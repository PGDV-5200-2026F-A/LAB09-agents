import requests

from base64 import b64encode
from io import BytesIO
from PIL import Image as PImage, ImageDraw as PImageDraw, ImageFont as PImageFont


def show_object_predictions(img, predictions):
  font = PImageFont.load_default(12)
  dimg = img.copy()
  draw = PImageDraw.Draw(dimg)

  for obj in predictions:
    label = obj["label"]
    (x0, y0, x1, y1) = tuple(obj["box"].values())
    draw.rectangle((x0,y0,x1,y1),
                   outline=(10, 220, 10),
                   width=2)
    draw.rectangle((x0,y0-6,x0+6*len(label),y0+6), fill=(0,0,0,160))
    draw.text((x0,y0-6), label, font=font)

  return dimg


def image_from_url(url):
  try:
    response = requests.get(url)
    response.raise_for_status()
    return PImage.open(BytesIO(response.content))
  except requests.exceptions.HTTPError:
    raise

def image_to_b64(img):
  buffered = BytesIO()
  img = img.convert("RGB")
  img.thumbnail((384, 384))
  img.save(buffered, format="JPEG")
  b64_str = b64encode(buffered.getvalue()).decode("utf-8")
  return { "data": b64_str, "mime_type": "image/jpeg" }

def image_path_to_b64(img_path):
  img = PImage.open(img_path).convert("RGB")
  return image_to_b64(img)

def image_url_to_b64(url):
  img = image_from_url(url)
  return image_to_b64(img)


# https://ai.google.dev/gemini-api/docs
def build_prompt(text, image_urls=None):
  input = [{
    "type": "text",
    "text": text
  }]

  if image_urls and type(image_urls) == str:
    image_urls = [image_urls]

  if image_urls and type(image_urls) == list:
    for url in image_urls:
      try:
        img_input = image_url_to_b64(url)
        img_input["type"] = "image"
        input.append(img_input)
      except requests.exceptions.HTTPError:
        pass

  return {
    "model": "gemini-3.5-flash-lite",
    "input": input,
    "generation_config": {
      "thinking_level": "minimal"
    }
  }
