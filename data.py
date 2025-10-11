import kaggle

kaggle.api.authenticate()

kaggle.api.dataset_download_files('ruizgara/socofing', path='data/', unzip=True)