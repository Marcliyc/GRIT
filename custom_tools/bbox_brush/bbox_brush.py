# from transformers import Tool
from smolagents import Tool
import base64
from PIL import Image, ImageDraw
import io
import json
def bbox_brush(image_base64, target_str, normalized_bboxs):
    # Decode the base64 image
    image = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image))
    draw = ImageDraw.Draw(image)
    try:
        if type(target_str) != str:
            for target_str in target_str:
                bbox = target_str.replace('(', '').replace(')', '').replace(']', '').replace('[', '').split(',')
                bboxes = [[int(x) for x in bbox]]
                assert len(bboxes[0]) == 4
                for bbox in bboxes:
                    if normalized_bboxs:
                        # x1, y1, x2, y2 are in 0,1000 range
                        bbox = [bbox[0]*image.width/1000, bbox[1]*image.height/1000, bbox[2]*image.width/1000, bbox[3]*image.height/1000]
                    draw.rectangle(bbox, outline='red', width=3)
            
        else:  
            bbox = target_str.replace('(', '').replace(')', '').replace(']', '').replace('[', '').split(',')
            bboxes = [[int(x) for x in bbox]]
            assert len(bboxes[0]) == 4
            for bbox in bboxes:
                if normalized_bboxs:
                    # x1, y1, x2, y2 are in 0,1000 range
                    bbox = [bbox[0]*image.width/1000, bbox[1]*image.height/1000, bbox[2]*image.width/1000, bbox[3]*image.height/1000]
                draw.rectangle(bbox, outline='red', width=3)
    except:
        return None, "Invalid bounding box format."


        
    # Encode image back to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_str, 'New image with bounding box attached: ' + str(bboxes)

# Test the function
# expression
# result = bbox_brush(expression)
# print(result)
print("bbox_brush function is ready to be used.")
# class BoundingboxBrush(Tool):
#     name = "bbox_brush"
#     description = (
#         "This is a tool that returns the a image that with bounding box drawn."
#     )

#     inputs = {"image_base64":{'type':'string', 
#                           'description':'The base64 image codings.',
#                           'content':'text'},
#               "bbox":{'type':'string',
#                       'description':'The bounding box coordinates in the format of x1,y1,x2,y2.',
#                       'content':'text'}
#               }
#     outputs = 'text'
#     output_type = "text"

#     def __call__(self, img_str: str, tool_call_query: str, normalized_bboxs: bool = False):
#         return bbox_brush(img_str, tool_call_query, normalized_bboxs)

class BoundingboxBrush(Tool):
    name = "bbox_brush"
    description = (
        "This is a tool that returns an image with a bounding box drawn on it."
    )

    # These keys (image_base64, bbox) MUST match the arguments in forward()
    inputs = {
        "image_base64": {
            'type': 'string', 
            'description': 'The base64 image coding.',
        },
        "bbox": {
            'type': 'string',
            'description': 'The bounding box coordinates in the format of x1,y1,x2,y2.',
        },
        'normalized_bboxs':{
            'type':'boolean',
            'description': 'Set to True if coordinates are 0-1000 relative, False if absolute pixels. Defaults to False.',
            'nullable': True
        }
    }
    
    # FIX 1: Change "text" to "string"
    output_type = "string"


    # FIX 2: Change __call__ to forward
    # FIX 3: Arguments match 'inputs' keys above
    def forward(self, image_base64: str, bbox: str, normalized_bboxs: bool = False):
        return bbox_brush(image_base64, bbox, normalized_bboxs)

