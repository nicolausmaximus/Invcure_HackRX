# PS4_Odd_Bit_Squad

## Introduction:
An interactive web application ideally suited for automating the process of fetching and classification of the text from an image in both handwritten as well as for printed invoices.

We have made the API and the web interface keeping in mind the flexibility and generalization of the problem statement. Our key features include
- An easy interface to interact with our API endpoint
- Our API is capable of detecting the text for both printed and handwritten invoices
- It returns the classified and categorized data along with the tagging of the data
- All of this made possible using openAI GPT-3 and open source OCR modules using pytesseract.
- A twin approach of digital forensics and CNN based fraud detection


Our application is loaded with features which ease the experience for the client and the insurance agent to easily access the data in the image:
Interactive, Responsive, Open AI GPT-3, CNN based fraud detection and Entity Tagging and Classification.


## API Endpoints:
<b>/                 </b>- Home page for our application <br>
<b>/authentication            </b>- Signs and authenticates an agent <br>
<b>/upload  </b> - Upload Page for the Images <br>
<b>/result  </b> - Prints the result of the Images and the relevant classified text from the images <br>

## Technology Stack:
 - Django
 - HTML, CSS, JS
 - Bootstrap4
 - Python
 - Open AI GPT-3
 - Mongo DB
 - CNN


### Installing in Windows

Clone the repository
```bash
git clone https://github.com/HackRx3/PS4_Odd_Bit_Squad.git
```
Make virtual environment for the project and activate it
```bash
python -m venv venv
venv\Scripts\Activate.ps1
```
Install the required packages in the virtual environment
```bash
pip install -r requirements.txt
```
### Running the Application
Apply the django migrations
```bash
python manage.py migrate
```
Collect the static files with modified timestamp
```bash
python manage.py collectstatic
```
Run the web server on the local machine
```bash
python manage.py runserver
```

## Screenshots
![1](https://user-images.githubusercontent.com/42286904/175799371-e05dc473-479d-4ac4-88bb-90d7a3d04dd8.png)
![image](https://user-images.githubusercontent.com/42286904/175799395-6dec861c-ec68-46aa-890a-f1906a6c2186.png)
![image](https://user-images.githubusercontent.com/42286904/175799399-047c0056-fff8-4344-bb4e-6a3f79bacb8d.png)
![image](https://user-images.githubusercontent.com/42286904/175799407-19058461-ff2b-4fe6-bcee-5c46998bd3b0.png)
![image](https://user-images.githubusercontent.com/42286904/175799412-40803610-447c-48ae-ba3d-df0363b7f689.png)
![image](https://user-images.githubusercontent.com/42286904/175799417-dc472408-ed5c-4968-84a3-d2136a76eead.png)
![image](https://user-images.githubusercontent.com/42286904/175799424-bc243b9c-6b37-4ccd-958c-73831558e2c8.png)


## Future Scope
- A homegrown handwritten recognition system

- Integrate the frontend with backend

- Improve the accuracy of the Tagging by using personalized datasets

## Contributors:

Team Name: Odd Bit Squad

* [Ayan Sadhukhan](https://github.com/ayan2809)
* [Ronit Sarkar](https://github.com/Codee0101)
* [Aniket Bansal](https://github.com/nicolausmaximus)


### Made at: Bajaj Hackatra 3.0
