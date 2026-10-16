import os
from distutils.dir_util import copy_tree

def create_folder(folder, access_rights):
    if not os.path.exists(folder):
        os.makedirs(folder, access_rights)
    else:
        print(f'Folder {folder} already exists!')



def create_paths(base_data, rater, subj, access_rights, train=True):
    subj_base = os.path.join(base_data, str(subj))
    subj_dest = os.path.join(base_data, f"Sliced_{rater}", str(subj))

    subj_dest = os.path.join(subj_dest, "train" if train else "test")

    flair_dest = os.path.join(subj_dest, "img", "out_flair", "img")
    t1_dest = os.path.join(subj_dest, "img", "out_t1", "img")
    t2_dest = os.path.join(subj_dest, "img", "out_t2", "img")
    pd_dest = os.path.join(subj_dest, "img", "out_pd", "img")
    msk1_dest = os.path.join(subj_dest, "msk", "out_lbl", "img")

    dest_folders = [flair_dest, t1_dest, t2_dest, pd_dest, msk1_dest]

    for folder in dest_folders:
        create_folder(folder, access_rights)

    return dest_folders

def get_paths(base_data, rater, subj, train=True):
    subj_dest = os.path.join(base_data, f"Sliced_{rater}", str(subj))
    subj_dest = os.path.join(subj_dest, "train" if train else "test")

    flair_dest = os.path.join(subj_dest, "img", "out_flair", "img")
    t1_dest = os.path.join(subj_dest, "img", "out_t1", "img")
    t2_dest = os.path.join(subj_dest, "img", "out_t2", "img")
    pd_dest = os.path.join(subj_dest, "img", "out_pd", "img")
    msk_dest = os.path.join(subj_dest, "msk", "out_lbl", "img")

    dest_folders = [flair_dest, t1_dest, t2_dest, pd_dest, msk_dest]

    return dest_folders


def create_folder_for_fold(tr_nets, dest, exp_name, fold, msk, access_rights):
    flair = os.path.join('out_flair', 'img')
    t1 = os.path.join('out_t1', 'img')
    t2 = os.path.join('out_t2', 'img')
    pd = os.path.join('out_pd', 'img')
    lbls = os.path.join('msk', 'out_lbl', 'img')

    trained_nets = os.path.join(tr_nets, exp_name, msk, exp_name + fold)
    dest_dir = os.path.join(dest, exp_name, msk, exp_name + fold, 'slices')
    train_dest = os.path.join(dest_dir, 'train')
    train_dest_im = os.path.join(train_dest, 'img')
    valid_dest = os.path.join(dest_dir, 'valid')
    valid_dest_im = os.path.join(valid_dest, 'img')

    # FLAIR, T1, T2 directories
    train_dest_im_flair = os.path.join(train_dest_im, flair)
    train_dest_im_t1 = os.path.join(train_dest_im, t1)
    train_dest_im_t2 = os.path.join(train_dest_im, t2)
    train_dest_im_pd = os.path.join(train_dest_im, pd)
    train_dest_msk = os.path.join(train_dest, lbls)

    valid_dest_im_flair = os.path.join(valid_dest_im, flair)
    valid_dest_im_t1 = os.path.join(valid_dest_im, t1)
    valid_dest_im_t2 = os.path.join(valid_dest_im, t2)
    valid_dest_im_pd = os.path.join(valid_dest_im, pd)
    valid_dest_msk = os.path.join(valid_dest, lbls)

    dest_dir_nets = os.path.join(dest, exp_name, msk, exp_name + fold)

    folders_train = [train_dest_im_flair, train_dest_im_t1, train_dest_im_t2,
                    train_dest_im_pd, train_dest_msk]
    folders_valid = [valid_dest_im_flair, valid_dest_im_t1, valid_dest_im_t2,
                    valid_dest_im_pd, valid_dest_msk]

    for folder in folders_train:
        create_folder(folder, access_rights)
    for folder in folders_valid:
        create_folder(folder, access_rights)
    create_folder(trained_nets, access_rights)

    folders = [folders_train, folders_valid, dest_dir_nets, trained_nets]

    return folders

def copy_data_for_fold(source_paths, dest_paths, train=True):
    for i in range(0, 5):
        copy_tree(source_paths[i], dest_paths[i])

    return
