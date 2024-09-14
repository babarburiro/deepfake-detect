# deepfake-detect
## A Project for Deepfake Detection Without Uploading a File. But by Screen Recording

### 1. Installation:
> pip install cmake
1. Install Visual Studio build tools from [here](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=15#).
2. In Visual Studio 2022 go to the Individual Components tab, Visual C++ Tools for Cmake, and check the checkbox under the "Compilers, build tools and runtimes" section.
> pip install dlib
> pip install -r requirements.txt

### 2. Running the application:
1. To run this application, just navigate to the main.py and type the following command in console/terminal.
> uvicorn main:app --reload

2. Open the browser and navigate to 
> 0.0.0.0:8000/



N.B: 
> The pt file is not uploaded due to heavy file size. Create a directory model and then put the checkpoint.pt file there. 