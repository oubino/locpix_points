# remove processed files
import os

for i in range(7):
    i = i + 1
    project_folder = f'semore_expts/task_{i}'

    print('Cleaning up')

    # clean up process folder check it first during debugging
    keep_files = ['file_map.csv', 'pre_filter.pt', 'pre_transform.pt']

    for fold in range(5):
        
        index = fold
        train_files = os.listdir(f'{project_folder}/processed/fold_{index}/train')
        val_files = os.listdir(f'{project_folder}/processed/fold_{index}/val')
        test_files = os.listdir(f'{project_folder}/processed/fold_{index}/test')
        # get list of folders to delete
        train_files = [os.path.join(f'{project_folder}/processed/fold_{index}/train', i) for i in train_files if i not in keep_files]
        val_files = [os.path.join(f'{project_folder}/processed/fold_{index}/val', i) for i in val_files if i not in keep_files]
        test_files = [os.path.join(f'{project_folder}/processed/fold_{index}/test', i) for i in test_files if i not in keep_files]

        for f in train_files:
            assert f.endswith('.pt')
        for f in val_files:
            assert f.endswith('.pt')
        for f in test_files:
            assert f.endswith('.pt')

        #for file in train_files:
           os.remove(file)
        for file in val_files:
           os.remove(file)
        for file in test_files:
           os.remove(file)
