from PIL import Image
from PIL.ExifTags import TAGS
# path to the image or video

def metadatacheck(imagename):
        
    # read the image data using PIL
    image = Image.open(imagename)
    
    # extract other basic metadata
    info_dict = {
    "Filename": image.filename,
    "Image Size": image.size,
    "Image Height": image.height,
    "Image Width": image.width,
    "Image Format": image.format,
    "Image Mode": image.mode,
    "Image is Animated": getattr(image, "is_animated", False),
    "Frames in Image": getattr(image, "n_frames", 1)
    }
    #   for label,value in info_dict.items():
    #    print(f"{label:25}: {value}")
    # extract EXIF data         
    exifdata = image.getexif()
    
    flag=0
    a=""
    b=""
    
    # iterating over all EXIF data fields
    for tag_id in exifdata:
        # get the tag name, instead of human unreadable tag 
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)
        # decode bytes 
        if isinstance(data, bytes):
                data = data.decode()
        print(f"{tag:25}: {data}")
        if({tag:25=="DateTime"}):
                flag=1;
        if({tag:25=="DateTimeOriginal"}):
                a={tag:25}
        if({tag:25=="DateTimeDigitized "}):
                b={tag:25}
        

        if(a!=b or flag==0):
                return("Edited")




