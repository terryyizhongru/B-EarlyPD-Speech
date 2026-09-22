import os
import torch
import numpy as np
import pandas as pd

class ParkinsonDataset(torch.utils.data.Dataset):
    def __init__(self, config, dataset_path, is_training=True):

        self.config = config
        if dataset_path.endswith('.csv'):
            self.dataset = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.tsv'):
            self.dataset = self._load_dataset(dataset_path)
        else:
            raise ValueError("Unsupported dataset format. Please provide a .csv or .tsv file.")

        print(f"Loaded dataset from {dataset_path} with {len(self.dataset)} samples.")
        if os.environ.get('PD_DEBUG_DATASET', '').strip() == '1':
            self.dataset.to_csv('debug_dataset.csv', index=False)
        # -- filtering tasks we are interested in, both for training and evaluation
        task_filter = [task['name'] for task in config.tasks]
        self.dataset = self.dataset[self.dataset['task_id'].isin(task_filter)]

        # -- filter participants with missing UPDRS data for perspective_splits dataset
        self._filter_perspective_participants(dataset_path)

        # -- handle label mapping and PD-only filtering
        self.target_label = getattr(config, 'target_label', 'label')
        self.class_mapping = getattr(config, 'class_mapping', None)
        self.pd_only = getattr(config, 'pd_only', False)

        # -- SSL (e.g., wav2vec) max-length truncation
        # Default wav2vec stride is ~20ms => ~50 frames/sec.
        self.ssl_fps = float(getattr(config, 'ssl_fps', 50.0))
        self.ssl_max_len_sec = getattr(config, 'ssl_max_len_sec', 10.0)
        if self.ssl_max_len_sec is None:
            self.ssl_max_len_frames = None
        else:
            self.ssl_max_len_sec = float(self.ssl_max_len_sec)
            self.ssl_max_len_frames = (
                int(round(self.ssl_max_len_sec * self.ssl_fps))
                if self.ssl_max_len_sec > 0
                else None
            )

        # Apply label transformations
        self._apply_label_mapping()

        # -- collecting informed speech feature metadata
        informed_metadata_ids = []
        informed_metadata_bounds = {}

        self.target_informed_idxs = {}
        self.target_informed_categories = {}
        for feature in self.config.features:
            feature_metadata_df = pd.read_csv(feature['metadata'])

            feature_metadata_ids = feature_metadata_df['feature_id'].tolist()
            self.target_informed_idxs[feature['name']] = feature_metadata_df['index_pos'].tolist()
            if 'category_id' in feature_metadata_df.columns:
                self.target_informed_categories[feature['name']] = feature_metadata_df['category_id'].tolist()
                if len(self.target_informed_categories[feature['name']]) != len(self.target_informed_idxs[feature['name']]):
                    raise ValueError(f"Feature {feature['name']} has different number of categories and indices.")

            start_bound = len(informed_metadata_ids)
            end_bound = start_bound + len(feature_metadata_ids)

            informed_metadata_ids += feature_metadata_ids
            informed_metadata_bounds[feature['name']] = (start_bound, end_bound)

        self.informed_metadata = (informed_metadata_ids, informed_metadata_bounds)

        # -- median and standard deviation of HC subjects in training for each type of feature.
        if is_training:
            # Use HC subjects (mapped_label == 0 if HC exists, otherwise skip HC-based normalization)
            hc_subjects = self.dataset[self.dataset['mapped_label'] == 0] if not self.pd_only else None
            if hc_subjects is not None and len(hc_subjects) > 0:
                self.feature_norm_stats = self.__compute_feature_norm_stats__(hc_subjects)
            else:
                # If no HC subjects or PD-only mode, use all subjects for normalization
                self.feature_norm_stats = self.__compute_feature_norm_stats__(self.dataset)
        else:
            # WARNING: For validation and test, these statistics are replaced in pipeline.py by those computed for training!
            self.feature_norm_stats = None

    def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load dataset from either CSV (native format) or TSV (NeuroVoz-style) and normalize to CSV schema."""
        _, ext = os.path.splitext(str(dataset_path))
        ext = ext.lower()
        if ext == '.tsv':
            tsv_df = pd.read_csv(dataset_path, sep='\t')
            return self._tsv_to_csv_schema(tsv_df)
        return pd.read_csv(dataset_path)

    @staticmethod
    def _tsv_to_csv_schema(tsv_df: pd.DataFrame) -> pd.DataFrame:
        """Convert a TSV with columns [ID, AUDIOFILE, DIAGNOSIS] into the expected CSV schema.

        Output columns:
        subject_id,sample_id,task_id,label,prosody,wav2vec,phonation,articulation,glottal
        """
        required = {'ID', 'AUDIOFILE', 'DIAGNOSIS'}
        missing = required.difference(set(tsv_df.columns))
        if missing:
            raise ValueError(f"TSV is missing required columns: {sorted(missing)}. Available columns: {list(tsv_df.columns)}")

        audiofile = tsv_df['AUDIOFILE'].astype(str)
        sample_id = audiofile.str.split('/').str[-1].str.replace(r'\.[Ww][Aa][Vv]$', '', regex=True)
        # Derive task_id from filename patterns.
        # NOTE: `"substr" in sample_id` is incorrect because `sample_id` is a Series.
        task_id = sample_id.str.split('_').str[-3]
        mask_sustained = sample_id.str.contains('SUSTAINED-VOWELS', regex=False)
        mask_ddk = sample_id.str.contains('DDK', regex=False)
        mask_sentences = sample_id.str.contains('SENTENCES', regex=False)

        task_id = task_id.where(~mask_sustained, 'SUSTAINED-VOWELS')
        task_id = task_id.where(~mask_sentences, 'SENTENCES')
        task_id = task_id.where(~mask_ddk, 'DDK')

        label = (tsv_df['DIAGNOSIS'].astype(str) == 'Parkinson').astype(int)

        # Feature paths are derived by replacing the AUDIOFILE path.
        prosody = audiofile.str.replace('audios_fortrain', 'speech_features/disvoice/prosody', regex=False).str.replace(
            r'\.[Ww][Aa][Vv]$', '.npz', regex=True
        )
        phonation = audiofile.str.replace('audios_fortrain', 'speech_features/disvoice/phonation', regex=False).str.replace(
            r'\.[Ww][Aa][Vv]$', '.npz', regex=True
        )
        articulation = audiofile.str.replace('audios_fortrain', 'speech_features/disvoice/articulation', regex=False).str.replace(
            r'\.[Ww][Aa][Vv]$', '.npz', regex=True
        )
        glottal = audiofile.str.replace('audios_fortrain', 'speech_features/disvoice/glottal', regex=False).str.replace(
            r'\.[Ww][Aa][Vv]$', '.npz', regex=True
        )
        wav2vec = audiofile.str.replace('audios_fortrain', 'speech_features/wav2vec/layer07', regex=False).str.replace(
            r'\.[Ww][Aa][Vv]$', '.npz', regex=True
        )

        out_df = pd.DataFrame({
            'subject_id': tsv_df['ID'].astype(str),
            'sample_id': sample_id,
            'task_id': task_id,
            'label': label,
            'prosody': prosody,
            'wav2vec': wav2vec,
            'phonation': phonation,
            'articulation': articulation,
            'glottal': glottal,
        })
        return out_df

    def _filter_perspective_participants(self, dataset_path):
        """Filter participants with missing UPDRS data for perspective_splits dataset"""
        # Check if this is a perspective_splits dataset and UPDRS is the target
        if 'perspective_splits' in dataset_path and getattr(self.config, 'target_label', 'label') == 'UPDRS':
            # Hardcoded list of participants with missing UPDRS data (converted from Screening_XXX format)
            participants_to_exclude = ['0098', '0009', '0116', '0181', '0182', '0137', '0214', '0112', '0082', '0019', '0090', '0034', '0050', '0014']

            original_len = len(self.dataset)
            print(f"Excluding {len(participants_to_exclude)} participants with missing UPDRS data from perspective_splits dataset")
            print(f"Participants to exclude: {participants_to_exclude}")

            # Filter out participants with missing UPDRS data
            # Ensure subject_id is string with zero-padding to match filter list format
            self.dataset['subject_id'] = self.dataset['subject_id'].astype(str).str.zfill(4)
            self.dataset = self.dataset[~self.dataset['subject_id'].isin(participants_to_exclude)]

            filtered_len = len(self.dataset)
            print(f"Dataset filtered: {original_len} -> {filtered_len} samples ({original_len - filtered_len} removed)")
        elif 'perspective_splits' in dataset_path:
            print(f"Perspective_splits dataset detected, but target_label is '{getattr(self.config, 'target_label', 'label')}', not 'UPDRS'. Skipping participant filtering.")

    def get_updrs_normalization_params(self):
        """Get UPDRS normalization parameters for denormalization"""
        if hasattr(self, 'updrs_min') and hasattr(self, 'updrs_max'):
            return self.updrs_min, self.updrs_max
        return None, None

    def __len__(self):
        return len(self.dataset)

    def _apply_label_mapping(self):
        """Apply label mapping based on target_label and class_mapping configuration"""

        # If using default binary classification with 'label' column, keep as is
        if self.target_label == 'label' and not self.class_mapping:
            self.dataset['mapped_label'] = self.dataset['label']
            return

        # For multi-class classification, use target_label column
        if self.target_label != 'label':
            if self.target_label not in self.dataset.columns:
                raise ValueError(f"Target label column '{self.target_label}' not found in dataset. Available columns: {list(self.dataset.columns)}")

            # Apply class mapping if provided
            if self.class_mapping:
                self.dataset['mapped_label'] = self.dataset[self.target_label].map(self.class_mapping)

                # Check for unmapped values
                unmapped = self.dataset[self.dataset['mapped_label'].isna()]
                if len(unmapped) > 0:
                    unmapped_values = unmapped[self.target_label].unique()
                    print(f"Warning: Found unmapped values in {self.target_label}: {unmapped_values}")
                    # Remove unmapped samples
                    self.dataset = self.dataset.dropna(subset=['mapped_label'])

                # Convert to appropriate type based on task
                task_type = getattr(self.config, 'task_type', 'classification')
                if task_type == 'regression':
                    self.dataset['mapped_label'] = self.dataset['mapped_label'].astype(float)
                else:
                    self.dataset['mapped_label'] = self.dataset['mapped_label'].astype(int)
            else:
                # Use target_label values directly
                self.dataset['mapped_label'] = self.dataset[self.target_label]
        else:
            # Using 'label' column but with class mapping
            if self.class_mapping:
                self.dataset['mapped_label'] = self.dataset['label'].map(self.class_mapping)
                # Convert to appropriate type based on task
                task_type = getattr(self.config, 'task_type', 'classification')
                if task_type == 'regression':
                    self.dataset['mapped_label'] = self.dataset['mapped_label'].astype(float)
                else:
                    self.dataset['mapped_label'] = self.dataset['mapped_label'].astype(int)
            else:
                self.dataset['mapped_label'] = self.dataset['label']

        # Apply PD-only filtering if enabled
        if self.pd_only:
            # Filter out HC subjects (typically class 0)
            original_len = len(self.dataset)
            self.dataset = self.dataset[self.dataset['label'] != 0]
            filtered_len = len(self.dataset)
            print(f"PD-only mode: filtered out {original_len - filtered_len} HC subjects, remaining: {filtered_len}")

            # Shift class labels down by 1 since we removed class 0 (HC)
            if hasattr(self.config, 'target_label') and self.config.target_label == 'H/Y':
                # For UPDRS regression, do not shift labels

                self.dataset['mapped_label'] = self.dataset['mapped_label'] - 1
                print("shifting H/Y labels.")

            print(f"PD-only mode: class distribution after shifting:")
            print(self.dataset['mapped_label'].value_counts().sort_index())

        # Apply MinMax normalization for UPDRS regression task
        if hasattr(self.config, 'task_type') and self.config.task_type == 'regression' and self.target_label == 'UPDRS':
            # UPDRS scores range from 0 to 132, normalize to 0-1
            self.updrs_min = 0.0
            self.updrs_max = 132.0
            print(f"UPDRS values before normalization: min={self.dataset['mapped_label'].min()}, max={self.dataset['mapped_label'].max()}")
            self.dataset['mapped_label'] = (self.dataset['mapped_label'] - self.updrs_min) / (self.updrs_max - self.updrs_min)
            print(f"Applied MinMax normalization for UPDRS: min={self.updrs_min}, max={self.updrs_max}")
            print(f"Normalized UPDRS range: [{self.dataset['mapped_label'].min():.3f}, {self.dataset['mapped_label'].max():.3f}]")
        else:
            # Set default values for non-UPDRS tasks
            self.updrs_min = None
            self.updrs_max = None

        print(f"Final class/label distribution:")
        if hasattr(self.config, 'task_type') and self.config.task_type == 'regression':
            print(f"Min: {self.dataset['mapped_label'].min():.3f}, Max: {self.dataset['mapped_label'].max():.3f}, Mean: {self.dataset['mapped_label'].mean():.3f}")
        else:
            print(self.dataset['mapped_label'].value_counts().sort_index())

    def __getitem__(self, index):
        sample = self.dataset.iloc[index]

        batch_sample = {}

        # -- subject and sample identification
        batch_sample['subject_id'] = sample['subject_id']
        batch_sample['sample_id'] = sample['sample_id']

        # -- speech-based features
        if self.config.model not in ['self_ssl']:
            batch_sample['informed_metadata'] = []
            for feature in self.config.features:

                if self.target_informed_categories != {}:

                    feature_data = []
                    for i, category_id in enumerate(self.target_informed_categories[feature['name']]):
                        # -- load the sample
                        feature_value = np.load(sample[category_id])['data'][:, self.target_informed_idxs[feature['name']][i]]
                        feature_value = feature_value.flatten()[0]
                        feature_data.append(feature_value)

                    feature_data = np.expand_dims(np.array(feature_data), axis=0)
                else:
                    feature_data = np.load(sample[feature['name']])['data'][:, self.target_informed_idxs[feature['name']]]

                feature_std = self.feature_norm_stats[feature['name']]['std']
                feature_median = self.feature_norm_stats[feature['name']]['median']
                batch_sample[feature['name']] = np.divide(feature_data - feature_median, feature_std, out=np.zeros(feature_data.shape), where=feature_std!=0)

        # -- ssl speech features
        if self.config.model not in ['self_inf']:
            ssl_data = np.load(sample[self.config.ssl_features])['data']
            ssl_data = self._truncate_ssl(ssl_data)
            ssl_median = self.feature_norm_stats[self.config.ssl_features]['median']
            ssl_std = self.feature_norm_stats[self.config.ssl_features]['std']

            batch_sample[self.config.ssl_features] = np.divide(ssl_data - ssl_median, ssl_std, out=np.zeros(ssl_data.shape), where=ssl_std!=0)

        # -- label ground truth
        batch_sample['label'] = sample['mapped_label']

        return batch_sample

    def _truncate_ssl(self, ssl_data: np.ndarray) -> np.ndarray:
        max_frames = getattr(self, 'ssl_max_len_frames', None)
        if max_frames is None:
            return ssl_data
        if not hasattr(ssl_data, 'shape') or len(getattr(ssl_data, 'shape', ())) == 0:
            return ssl_data
        if ssl_data.shape[0] <= max_frames:
            return ssl_data
        return ssl_data[:max_frames, ...]

    def collate_fn(self, batch):
        pad_batch = {}

        for key in batch[0].keys():
            if key in [self.config.ssl_features]:
                pad_batch[key] = [torch.Tensor(batch_sample[key]) for batch_sample in batch]
                # -- computing mask
                pad_batch['ssl_lengths'] = [ssl_sample.shape[0] for ssl_sample in pad_batch[key]]
                pad_batch['mask_ssl'] = (~self.__make_pad_mask__(pad_batch['ssl_lengths'])[:, None, :])
            else:
                pad_batch[key] = [batch_sample[key] for batch_sample in batch]

            if key not in ['subject_id', 'sample_id', 'group', 'task_id']:
                if key in [self.config.ssl_features]:
                    pad_batch[key] = torch.nn.utils.rnn.pad_sequence(pad_batch[key], batch_first=True).type(torch.float32)
                elif key not in ['mask_ssl']:
                    # Handle label data type based on task type
                    if key == 'label':
                        # For regression tasks, use float32; for classification, use int64
                        task_type = getattr(self.config, 'task_type', 'classification')
                        if task_type == 'regression':
                            pad_batch[key] = torch.Tensor(np.array(pad_batch[key])).type(torch.float32)
                        else:
                            pad_batch[key] = torch.Tensor(np.array(pad_batch[key])).type(torch.int64)
                    else:
                        # For other keys (except ssl_lengths which should be int64)
                        pad_batch[key] = torch.Tensor(np.array(pad_batch[key])).type(torch.float32 if key not in ['ssl_lengths'] else torch.int64)

        return pad_batch

    def __make_pad_mask__(self, lengths):
        bs = int(len(lengths))
        maxlen = int(max(lengths))

        seq_range = torch.arange(0, maxlen, dtype=torch.int64)
        seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
        seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
        mask = seq_range_expand >= seq_length_expand

        return mask

    def __compute_feature_norm_stats__(self, hc_dataset):
        features_ids = [feature['name'] for feature in self.config.features] + [self.config.ssl_features]
        feature_norm_stats = {feature_id:{'median': 0.0, 'std': 1.0} for feature_id in features_ids}

        # -- statistics for informed speech features
        if self.config.model not in ['self_ssl']:
            for feature in self.config.features:
                if self.target_informed_categories != {}:

                    samples = []
                    for i, category_id in enumerate(self.target_informed_categories[feature['name']]):
                        # -- load the sample
                        samples1 = []
                        for sample_path in hc_dataset[category_id].tolist():
                            feature_value = np.load(sample_path)['data'][:, self.target_informed_idxs[feature['name']][i]]
                            # 确保是标量或一维数组，然后添加到特征列表
                            feature_value = feature_value.flatten()[0]
                            samples1.append(feature_value)
                        samples.append(np.array(samples1))
                        # -- concatenate the samples

                    samples = np.stack(samples, axis=1)
                    samples = np.expand_dims(samples, axis=1)

                else:
                    samples = np.array([
                        np.load(sample_path)['data'][:, self.target_informed_idxs[feature['name']]]
                        for sample_path in hc_dataset[feature['name']].tolist()
                        ])

                feature_norm_stats[feature['name']]['median'] = np.median(samples, axis=0)
                feature_norm_stats[feature['name']]['std'] = np.std(samples, axis=0)

        # -- statistics for SSL speech features
        if self.config.model not in ['mlp_inf', 'self_inf']:
            samples = np.concatenate([
                self._truncate_ssl(np.load(sample_path)['data'])
                for sample_path in hc_dataset[self.config.ssl_features].tolist()
            ], axis=0 )

            feature_norm_stats[self.config.ssl_features]['median'] = np.median(samples, axis=0)
            feature_norm_stats[self.config.ssl_features]['std'] = np.std(samples, axis=0)

        return feature_norm_stats
