import torch
from einops.layers.torch import Reduce
from .multiheaded_attention import MultiHeadedAttention
import torch.nn as nn


def create_coral_targets(labels, num_classes):
    """
    Create CORAL targets from class labels.
    For a label y in range [0, num_classes-1], create a target vector of length num_classes-1
    where target[i] = 1 if i < y, else 0.
    """

    batch_size = labels.size(0)
    targets = torch.zeros(batch_size, num_classes - 1, device=labels.device, dtype=torch.float32)

    for i in range(batch_size):
        label = labels[i].item()
        for j in range(num_classes - 1):
            if j < label:
                targets[i, j] = 1.0
            else:
                targets[i, j] = 0.0

    return targets


def coral_predict(outputs):
    """
    Convert CORAL outputs to class predictions.
    Sum the cumulative probabilities to get the predicted class.
    """
    # outputs shape: (batch_size, num_classes-1)
    probas = torch.sigmoid(outputs)
    predictions = torch.sum(probas > 0.5, dim=1)
    return predictions


class RECAPD_Model(torch.nn.Module):
    """SSL Embedding & Temporal Cross-Attention Model.
    """

    def __init__(self, config):
        super(RECAPD_Model, self).__init__()

        self.config = config
        self.attn_type = self.config.model

        # Get task configuration
        self.task_type = getattr(config, 'task_type', 'classification')
        self.target_label = getattr(config, 'target_label', 'label')

        # -- computing informed-based speech features input dimension
        informed_input_dim = 0
        for feature in self.config.features:
            informed_input_dim += feature['input_dim']

        self.feature_nets = torch.nn.ModuleDict()
        for feat in config.features:
            name = feat['name']
            in_dim = feat['input_dim']
            # linear + activation to map to D
            self.feature_nets[name] = self.make_three_layer_net(in_dim, self.config.ssl_features_conf['input_dim'])
            # print(self.feature_nets[name])
        # -- model architecture setup
        # 'cross_embed': [DxT] · [TxF] --> [DxF] --> softmax([FxD]) · [DxT] --> [FxT] --> mean([FxT]) --> [F] --
        #                                                                                                                   --> Classification
        # 'cross_time':  [TxD] · [DxF] --> [TxF] --> softmax([FxT]) · [TxD] --> [FxD] --> mean([FxD]) --> [F] --
        self.query_dim = self.config.ssl_features_conf['input_dim']
        self.key_dim = self.config.ssl_features_conf['input_dim']
        self.value_dim = self.config.ssl_features_conf['input_dim']
        self.D = self.config.ssl_features_conf['input_dim']

        self.cross_attn = MultiHeadedAttention(
            query_dim=self.D,
            key_dim=self.D,
            value_dim=self.D,
            num_heads=config.model_conf['num_heads'],
            dropout_rate=config.model_conf['dropout'],
            attn_type='new',
        )
        # # -- embedding cross attention
        # self.embed_mha = MultiHeadedAttention(
        #     query_dim=self.query_dim,
        #     key_dim=self.key_dim,
        #     value_dim=self.value_dim,
        #     num_heads=self.config.model_conf['num_heads'],
        #     dropout_rate=self.config.model_conf['dropout'],
        #     attn_type='cross_embed',
        # )

        # self.cross_attn = MultiHeadedAttention(
        #     query_dim=self.query_dim,
        #     key_dim=self.key_dim,
        #     value_dim=self.value_dim,
        #     num_heads=config.model_conf['num_heads'],
        #     dropout_rate=config.model_conf['dropout'],
        #     attn_type='cross_time',  # new type if necessary
        # )

        # # -- temporal cross attention
        # self.time_mha = MultiHeadedAttention(
        #     query_dim=self.query_dim,
        #     key_dim=self.key_dim,
        #     value_dim=self.value_dim,
        #     num_heads=self.config.model_conf['num_heads'],
        #     dropout_rate=self.config.model_conf['dropout'],
        #     attn_type='cross_time',
        # )

        # -- classifier
        if self.task_type == 'regression':
            # For regression, output single value
            if getattr(config, 'target_label', 'label') == 'UPDRS':
                # For UPDRS regression, add sigmoid to ensure 0-1 output
                self.classifier = torch.nn.Sequential(
                    torch.nn.LayerNorm(self.key_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(self.key_dim, 1, bias=False),
                    torch.nn.Sigmoid(),  # Ensure output is in [0, 1] range
                )
            else:
                # For other regression tasks, keep original structure
                self.classifier = torch.nn.Sequential(
                    torch.nn.LayerNorm(self.key_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(self.key_dim, 1, bias=False),
                )
        elif self.task_type == 'ordinal_regression':
            # For ordinal regression (CORAL), output cumulative logits
            self.classifier = torch.nn.Sequential(
                torch.nn.LayerNorm(self.key_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(self.key_dim, self.config.num_classes - 1, bias=False),
            )
        else:
            # For classification, output logits for each class
            self.classifier = torch.nn.Sequential(
                torch.nn.LayerNorm(self.key_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(self.key_dim, self.config.num_classes, bias=False),
            )
        print('last layer:',self.classifier)
        # Computing Loss Function
        if self.task_type == 'regression':
            self.loss_criterion = torch.nn.MSELoss(reduction='mean')
        elif self.task_type == 'ordinal_regression':
            # CORAL loss will be computed in forward pass
            self.loss_criterion = None
        elif config.training_settings['loss_criterion'] == 'cross_entropy':
            self.loss_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        else:
            raise ValueError(f'unknown loss criterion {config.training_settings["loss_criterion"]}')

    def coral_loss(self, logits, labels):
        """
        Compute CORAL (COnsistent RAnk Logits) loss for ordinal regression.

        Args:
            logits: (B, num_classes-1) cumulative logits
            labels: (B,) ordinal labels
        """
        cumulative_targets = create_coral_targets(labels, self.config.num_classes)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, cumulative_targets)
        return loss

    def forward(self, batch):
        model_output = {}
        # import pdb; pdb.set_trace()
        B, T, D = batch[self.config.ssl_features].shape

        tokens = []
        for feat in self.config.features:
            name = feat['name']
            x = batch[name]           # (B, 1, in_dim)
            x = x.squeeze(1)          # (B, in_dim)
            tok = self.feature_nets[name](x)  # (B, D)
            tokens.append(tok)

        tokens = torch.stack(tokens, dim=1)  # (B, K, D)

        Q = batch[self.config.ssl_features]  # (B, T, D)
        # cross attention: Q x (K,D) tokens
        out = self.cross_attn(Q, tokens, tokens, mask=batch['mask_ssl'])  # (B, T, D)
        # aggregate over time
        repr = Reduce('b n d -> b d', 'mean')(out)                               # (B, D)

        # forward through classifier
        logits = self.classifier(repr)
        if self.task_type == 'regression':
            logits = logits.squeeze(-1)  # (B,) for regression

        #========================
        # build model_output
        model_output['subject_id'] = batch['subject_id']
        model_output['sample_id']  = batch['sample_id']
        model_output['embeddings'] = repr
        model_output['logits']     = logits

        # Handle different task types
        if self.task_type == 'regression':
            # For UPDRS regression, use normalized values for loss but provide denormalized for metrics
            if getattr(self.config, 'target_label', 'label') == 'UPDRS':
                # Keep normalized predictions and labels for loss calculation
                normalized_preds = logits
                normalized_labels = batch['label'].float()

                # Denormalize for metrics/reference
                updrs_min, updrs_max = 0.0, 132.0
                denorm_preds = logits * (updrs_max - updrs_min) + updrs_min
                denorm_labels = batch['label'].float() * (updrs_max - updrs_min) + updrs_min

                # Use normalized values for loss computation (scaled by 10)
                model_output['loss'] = self.loss_criterion(normalized_preds, normalized_labels)

                # Provide both normalized and denormalized values
                model_output['normalized_preds'] = normalized_preds  # For loss calculation
                model_output['normalized_labels'] = normalized_labels  # For loss calculation
                model_output['preds'] = denorm_preds  # Denormalized predictions for reference metrics
                model_output['labels'] = denorm_labels  # Denormalized labels for reference metrics
            else:
                model_output['preds'] = logits  # For other regression tasks, prediction is the logit value
                model_output['labels'] = batch['label'].float()  # Always use processed 'label'
                model_output['loss'] = self.loss_criterion(logits, model_output['labels'])
        elif self.task_type == 'ordinal_regression':
            # For ordinal regression, use CORAL approach
            model_output['preds'] = coral_predict(logits)
            model_output['labels'] = batch['label'].long()  # Always use processed 'label'
            model_output['probs'] = torch.sigmoid(logits)  # Cumulative probabilities
            model_output['loss'] = self.coral_loss(logits, model_output['labels'])
        else:
            # Classification
            model_output['probs'] = torch.nn.functional.softmax(logits, dim=-1)
            model_output['preds'] = logits.argmax(dim=-1)
            model_output['labels'] = batch['label'].long()  # Always use processed 'label'
            model_output['loss'] = self.loss_criterion(logits, model_output['labels'])

        model_output['attn_scores'] = self.cross_attn.attn_scores
        return model_output


    def make_three_layer_net(self, in_dim: int, target_dim: int = 1024,
                            mid1: int = 128, mid2: int = 512,
                            use_norm: bool = True, p_drop: float = 0.1):
        """
        3‑层 MLP: in_dim → mid1 → mid2 → target_dim
        每层 SiLU，选配 LayerNorm + Dropout
        """
        layers = []

        # ① in_dim → mid1
        layers.append(nn.Linear(in_dim, mid1, bias=False))
        if use_norm:
            layers.append(nn.LayerNorm(mid1))
        layers.append(nn.SiLU())
        if p_drop > 0:
            layers.append(nn.Dropout(p_drop))

        # ② mid1 → mid2
        layers.append(nn.Linear(mid1, mid2, bias=False))
        if use_norm:
            layers.append(nn.LayerNorm(mid2))
        layers.append(nn.SiLU())
        if p_drop > 0:
            layers.append(nn.Dropout(p_drop))

        # ③ mid2 → target_dim
        layers.append(nn.Linear(mid2, target_dim, bias=False))
        layers.append(nn.SiLU())          # 若要保持线性输出，可把这一行删掉

        return nn.Sequential(*layers)
