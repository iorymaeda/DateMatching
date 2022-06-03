import os
import shutil
from contextlib import suppress

import torch
from lxml import html
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


import utils


def get_name(body: html.HtmlElement) -> str|None:
    with suppress(Exception):
        name = body.find_class('from_name')[0].text
        name = name.replace('\n', '').strip()
        return name

def get_photo(body: html.HtmlElement) -> str|None:
    with suppress(Exception):
        div = body.find_class('media_wrap')[0]
        a = div.getchildren()[0]
        photo = a.attrib['href']
        return photo

def get_text(body: html.HtmlElement, text="") -> str|None:
    with suppress(Exception):
        div = body.find_class('text')[0]
        for t in div.itertext():
            text+= t
            
        text = text.replace('\n', '', 2).strip()
        return text

def process_photo(photo_path:str, saved_path:str):
    save_embs = True
    if photo_path in embeder.path_id:
        uid = embeder.path_id[photo_path]
        if os.path.exists(f'{saved_path}/{uid}.png'):
            return
        
        else:
            save_embs = False

    img = Image.open(photo_path)
    _, e = photo_path.split('.')
    if e.lower() == 'png':
        img = img.convert('RGB')
        
    img_cropped = mtcnn(img, save_path='tmp.png')
    
    # Check for founded face
    if img_cropped is not None:
        img_cropped = img_cropped.unsqueeze(0)
    
        if save_embs:
            embs = [model(img_cropped) for model in embeds_models]
            embeder[photo_path] = embs

        uid = embeder.path_id[photo_path]
        shutil.move('tmp.png', f'{saved_path}/{uid}.png')


def procces_markup(src:str, dst:str):
    if not src.endswith('/'):
        src+= '/'
        
    if dst.endswith('/'):
        dst = dst[:-1]
        
    for file in os.listdir(src):
        photo_path = src + file
        process_photo(photo_path, dst)


if __name__ == "__main__":
    mtcnn = MTCNN()
    model1 = InceptionResnetV1(pretrained='vggface2').eval()
    model2 = InceptionResnetV1(pretrained='casia-webface').eval()

    embeds_models = [model1, model2]
    embeder = utils.PhotoEmbedingStorage()


    messages = []
    for catalog in os.listdir('tg_data'):
        
        for file in os.listdir('tg_data/' + catalog):
            if '.html' in file:

                with open(f"tg_data/{catalog}/{file}", "r", encoding='utf-8') as f:
                    _html = f.read()

                tree = html.fromstring(_html)
                history = tree.xpath("/html/body/div/div[2]/div")[0]
                history = history.find_class('message default clearfix')

                for message in history:
                    body = message.find_class('body')[0]
                        
                    photo = get_photo(body) 
                    photo = f"{catalog}/{photo}" if photo else None
                    messages+= [{
                        'name': get_name(body),
                        'text': get_text(body),
                        'photo': photo,
                    }]

    with torch.no_grad():
        for current, previous in zip(messages[1:], messages[:-1]):
            if previous['photo']:
                photo_path = 'tg_data/' + previous['photo']

                if current['text'] in ['💌', '❤️', '👍']:
                    saved_path = 'data/target'

                elif current['text'] in ['👎']:
                    saved_path = 'data/opposite'

                else:
                    continue
            
                process_photo(photo_path, saved_path)


        procces_markup('markup/test_target',  "data/test_target/")
        procces_markup('markup/test_opposite',  "data/test_opposite/")
        procces_markup('markup/target',  "data/markup_target/")
        procces_markup('markup/opposite',  "data/markup_opposite/")

    embeder.save('emb storage.pkl')