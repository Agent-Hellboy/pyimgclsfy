**pyimgclsfy**
==============



      Library for image classification 
      
      

**Installation**
----------------


    git clone https://github.com/princekrroshan01/pyimgclsfy/
    cd pyimgclsfy
    virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt
      
**General Info**
----------------

    pass

**Prerequisites**
-----------------
    
    pass


**using**
---------

```

from imagenet import ImageNet

# pass image url like monkey.jpeg into the class 
# if image is in root of your project directory else whole path
obj=ImageNet(image_url)

# returns label of the image
obj.get_label()


```


**Contributing**
----------------

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.
