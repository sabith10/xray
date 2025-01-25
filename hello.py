from PIL import Image
from flask import Flask, send_file


app = Flask(__name__)


@app.route('/')
def main():
    folder_path = 'chest-xrays/image'
    im = Image.open(f"{folder_path}/1000.png")
    return send_file(im)

if __name__ == "__main__":
    app.run(debug=True)
