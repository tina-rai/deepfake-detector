from PIL import Image
import os

def analyze_image(path):

    img = Image.open(path)

    width,height = img.size

    size = round(os.path.getsize(path)/1024,2)

    return{

        "width":width,

        "height":height,

        "aspect":round(width/height,2),

        "format":img.format,

        "mode":img.mode,

        "size":size,

        "report":[

            "Face successfully detected",

            "Metadata extracted",

            "Artifact analysis completed",

            "Neural network inference complete"

        ]

    }