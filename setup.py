from setuptools import setup, find_packages 
setup( 
	name = "rselector", 
	version = "1.0", 
	keywords = ("pip", "SICA","particle selection","CNN"), 
	description = "An CNN particle selector", 
	long_description = "", 
	license = "MIT Licence", 
	url = "https://github.com/LiangjunFeng/SICA", 
	author = "duxuan", 
	author_email = "", 
	packages = find_packages(), 
	include_package_data = True, 
	platforms = "any", 
	install_requires = ["numpy","tensorflow","keras","cv2","mrcz","scipy","progressbar","zignor","keras_resnet","cython","argparse"]  )

