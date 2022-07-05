import os
import sys
import time
import pathlib
import argparse
from concurrent.futures import ThreadPoolExecutor

from instagrapi import Client

import config


def download(medias: list):
    num_batches = len(medias)//config.batch_size + int(len(medias)%config.batch_size != 0)
    for b in range(num_batches):
        time_start = time.time()
        
        tasks = []
        for media in medias[b*config.batch_size:(b+1)*config.batch_size]:
            if media.media_type == 1:
                task =  executor.submit(cl.photo_download, media.pk, folder=save_path)
                tasks.append(task)
            elif media.media_type == 8:
                task = executor.submit(cl.album_download, media.pk, folder=save_path)
                tasks.append(task)
            
            
        [t.result() for t in tasks]
        
        passed_time = time.time() - time_start
        if passed_time < config.timeout:
            time.sleep(config.timeout - passed_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse instagram profile')
    base_group1 = parser.add_argument_group('Login info')
    base_group1.add_argument('-login', '-lg', type=str, help='login to your account, if none will get from environ as `InstaLogin`', default=os.environ.get("InstaLogin"))
    base_group1.add_argument('-password','-ps', type=str, help='password to your account, if none will get from environ as `InstaPassword`', default=os.environ.get("InstaPassword"))

    base_group2 = parser.add_argument_group('Parse info')
    base_group2.add_argument('-names', '-n', type=str, help='user names to parse', nargs='+', action='store')
    base_group2.add_argument('-out', '-o', type=str, help="folder to save ", default="")
    base_group2.add_argument('-batch_size', '-bs', type=str, help="number of media to download at a time", default=config.batch_size)
    base_group2.add_argument('-timeout', '-to', type=str, help="time between batches downloads", default=config.timeout)
    base_group2.add_argument('-max', type=str, help="max media amount to download", default=config.max_media_amount)


    args = parser.parse_args()
    config.max_media_amount = args.max
    config.batch_size = args.batch_size
    config.timeout = args.timeout
    if args.out:
        args.out += '/'

    for user_name in args.names:
        save_path = pathlib.Path(f'{args.out}{user_name}/')
        save_path.mkdir(exist_ok=True)


    cl = Client()
    cl.login(args.login, args.password)
    executor = ThreadPoolExecutor(max_workers=config.batch_size)

    for user_name in args.names:
        user_id = cl.user_id_from_username(user_name)
        medias = cl.user_medias(user_id, config.max_media_amount)
        save_path = pathlib.Path(f'{args.out}{user_name}/')
        download(medias)

    sys.exit(0)