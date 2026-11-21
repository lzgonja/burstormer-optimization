"""Download and unpack the pre-processed BurstSR dataset (train and val).

The original host (data.vision.ee.ethz.ch/bhatg/BurstSRChallenge/) returns 404 since at least 2026-11. The files are
fetched from the Google Drive folder linked in the DBSR README (github.com/goutamgmb/deep-burst-sr):
https://drive.google.com/drive/folders/1A8FytFrYP-y7pZZhfFxVRzEPwcKl-Cp7

Usage (from `Burst Super-resolution/`):
    python scripts/download_burstsr_dataset.py .            # train + val
    python scripts/download_burstsr_dataset.py . --skip_val  # train only
"""
import os
import shutil
import zipfile
import argparse
import urllib.request

DRIVE_IDS = {
    'train_00': '1sUlIINqiY3Oz26wcdwUv_JQdQLyi1sfy',
    'train_01': '1mHIoX_20isi9JgaTzo3WdE2_VzDt0v3z',
    'train_02': '1-VCMeu-cf9SrkBa2CIUVDrYBNtJmbGC_',
    'train_03': '13K9MONzIbnoU-297EJaW4tYMyBMDHvLy',
    'train_04': '10X8-D-LOjvSj9U7hqXSSC2rF5VyUq5ZK',
    'train_05': '1FND7LrMci7BJhLORxLZWa0gn99F3IuGU',
    'train_06': '1H93sF2smGtFFiPLMwRSJ4BN3O5ruIJ4c',
    'train_07': '1fu0aIxIbYwpMk1QKM4C5CrKjTgPpszbm',
    'train_08': '1UKwMCh4koC5781uVLYXGVoH4N7EuC6JL',
    'val': '1Qjx1jCMhdabkCedoJP704cGyM3wo8QiH',
}
URL = 'https://drive.usercontent.google.com/download?id={}&export=download&confirm=t'


def fetch(name, out_dir):
    dst = os.path.join(out_dir, name + '.zip')
    if os.path.isfile(dst):
        return dst
    print('Downloading', name, flush=True)
    tmp = dst + '.part'
    urllib.request.urlretrieve(URL.format(DRIVE_IDS[name]), tmp)
    if not zipfile.is_zipfile(tmp):
        raise RuntimeError('{} is not a zip file; Google Drive may have returned a quota or warning page'.format(tmp))
    os.rename(tmp, dst)
    return dst


def download_burstsr_dataset(download_path, skip_val=False, keep_zips=False):
    out_dir = os.path.join(download_path, 'burstsr_dataset')
    os.makedirs(out_dir, exist_ok=True)
    train_parts = ['train_{:02d}'.format(i) for i in range(9)]

    if not os.path.isdir(os.path.join(out_dir, 'train')):
        for name in train_parts:
            fetch(name, out_dir)

        for name in train_parts:
            print('Unpacking', name, flush=True)
            with zipfile.ZipFile(os.path.join(out_dir, name + '.zip')) as z:
                z.extractall(out_dir)

        # Each zip unpacks to train_XX/<burst>; move all bursts into one train/ directory.
        os.makedirs(os.path.join(out_dir, 'train_tmp'), exist_ok=True)
        for name in train_parts:
            part_dir = os.path.join(out_dir, name)
            for b in os.listdir(part_dir):
                if os.path.isdir(os.path.join(part_dir, b)):
                    shutil.move(os.path.join(part_dir, b), os.path.join(out_dir, 'train_tmp', b))
            shutil.rmtree(part_dir)
        os.rename(os.path.join(out_dir, 'train_tmp'), os.path.join(out_dir, 'train'))

        if not keep_zips:
            for name in train_parts:
                os.remove(os.path.join(out_dir, name + '.zip'))
    else:
        print('train/ already present, skipping')

    if not skip_val and not os.path.isdir(os.path.join(out_dir, 'val')):
        zpath = fetch('val', out_dir)
        print('Unpacking val', flush=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out_dir)

    print('Done:', out_dir)


def main():
    parser = argparse.ArgumentParser(description='Downloads and unpacks the BurstSR dataset')
    parser.add_argument('path', type=str, help='Directory in which burstsr_dataset/ is created')
    parser.add_argument('--skip_val', action='store_true')
    parser.add_argument('--keep_zips', action='store_true')
    args = parser.parse_args()
    download_burstsr_dataset(args.path, args.skip_val, args.keep_zips)


if __name__ == '__main__':
    main()
