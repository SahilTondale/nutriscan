import os
import numpy as np
import traceback
from PIL import Image
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import IngredientAnalysis
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from paddleocr import PaddleOCR
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Lazy initialization of API keys - only validate when actually needed
def get_api_keys():
    langchain_key = os.getenv("LANGCHAIN_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not groq_key:
        raise ValueError("GROQ_API_KEY environment variable is required")
    
    if not langchain_key:
        raise ValueError("LANGCHAIN_API_KEY environment variable is required")
    
    os.environ["LANGCHAIN_API_KEY"] = langchain_key
    os.environ["GROQ_API_KEY"] = groq_key
    
    return groq_key, langchain_key

# Initialize model only when needed
model = None
parser = StrOutputParser()

def get_model():
    global model
    if model is None:
        get_api_keys()  # This will validate keys
        # UPDATED: Changed from decommissioned 'llama3-8b-8192' to 'llama-3.3-70b-versatile'
        model = ChatGroq(model="llama-3.3-70b-versatile")
    return model

# Create the prompt template
system_template = '''As a health analysis expert, analyze {category} ingredients from this list: {list_of_ingredients} while considering with STRICT adherence to:
- User allergies: {allergies}
- User medical history: {diseases}

**IMPORTANT: Only proceed with analysis if valid ingredients are detected and category is appropriate. If no valid ingredients are found or category is incorrect, respond with: "Since no valid ingredients are detected for this category, there are no risks specific to the user's profile."**

**Structured Analysis Framework:**

1. **Key Ingredient Analysis** (Focus on 4-5 most significant):
    For each impactful ingredient:
    - Primary use in {category}
    - Benefits (if any)
    - Risks (prioritize allergy/condition conflicts)
    - Safety status vs daily limits

2. **Personalized Health Impact** ⚠️:
    - Top 3 risks specific to user's profile :
      - Frequency of use
      - Quantity in product
      - Medical history interactions
      
3. **Should Take or Not 🔍:
    - Ingredients list which are dangerous for user's allergies and conditions :
    - Final recommendation, Should user take this product or not: 
    
4. **Smart Alternatives** 💡:
    - 2-3 safer options avoiding flagged ingredients
    - Benefits for user's specific needs
    - Category-appropriate substitutions

Format concisely using bullet points, warning symbols(❗), and prioritize medical-critical information. Ignore unrecognized/unimportant ingredients.'''

prompt_template = ChatPromptTemplate.from_messages([("system", system_template)])

# Ensure the user is logged in
def home(request):
    return render(request, "home.html")

@login_required
def upload(request):
    return render(request, "NutriScan_app/upload.html")

class OCRReader:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(OCRReader, cls).__new__(cls, *args, **kwargs)
            cls._instance.reader = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
        return cls._instance

    def read_text(self, img):
        # PaddleOCR returns different format: [[[bbox], (text, confidence)]]
        results = self.reader.ocr(img, cls=True)
        # Extract only text from PaddleOCR results
        text_list = []
        if results and results[0]:
            for line in results[0]:
                if len(line) > 1:
                    text_list.append(line[1][0])  # line[1][0] contains the text
        return text_list

ocr_reader = OCRReader()

@csrf_exempt
@login_required 
def analyze_ingredients(request):
    if request.method == "POST":
        image = request.FILES.get("image")
        category = request.POST.get("category")

        if image and category:
            try:
                # 1. Create the DB record INSIDE the try block
                analysis = IngredientAnalysis.objects.create(
                    user=request.user, 
                    category=category, 
                    image=image,
                    result=""
                )

                img = Image.open(image)
                img = np.array(img)

                # 2. Run OCR
                results = ocr_reader.read_text(img)
                text_only = [item for item in results if isinstance(item, str)]
                
                print(f"OCR Results: {text_only}")
                
                # 3. Handle Medical History Safely
                try:
                    if hasattr(request.user, 'medicalhistory'):
                        mh = request.user.medicalhistory
                        allergies = mh.allergies.split(',') if mh.allergies else ["No allergy"]
                        diseases = mh.diseases.split(',') if mh.diseases else ["No disease"]
                    else:
                        allergies = ["No allergy"]
                        diseases = ["No disease"]
                except Exception as e:
                    print(f"Medical history warning: {e}")
                    allergies = ["No allergy"]
                    diseases = ["No disease"]
                
                ingredients_text = ", ".join(text_only) if text_only else "No text detected"
                
                # 4. Run LLM
                model_instance = get_model()
                chain = prompt_template | model_instance | parser
                llm_response = chain.invoke(
                    {
                        "list_of_ingredients": ingredients_text, 
                        "category": category, 
                        "allergies": ", ".join(allergies), 
                        "diseases": ", ".join(diseases)
                    }
                )
                
                analysis.result = llm_response
                analysis.save()

                return JsonResponse({"result": llm_response, "analysis_id": analysis.id})

            except Exception as e:
                # This prints the FULL error to your terminal so we can see it
                print("------------- CRITICAL PROCESSING ERROR -------------")
                traceback.print_exc() 
                print("-----------------------------------------------------")
                
                if 'analysis' in locals():
                    analysis.delete()
                
                return JsonResponse({"error": f"Processing failed: {str(e)}"}, status=500)

        return JsonResponse({"error": "Invalid input"}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)

def history(request):
    user_analyses = IngredientAnalysis.objects.filter(user=request.user).order_by("-timestamp")
    return render(request, "history.html", {"user_analyses": user_analyses})

def analysis_detail(request, analysis_id):
    analysis = IngredientAnalysis.objects.get(id=analysis_id, user=request.user)
    return render(request, "analysis_detail.html", {"analysis": analysis})

def register(request):
    if request.method == "POST":
        username = request.POST["username"]
        email = request.POST["email"]
        password = request.POST["password"]
        repeat_password = request.POST["repeatPassword"]

        if password != repeat_password:
            return render(request, "register.html", {"error_message": "Passwords don't match"})

        if User.objects.filter(username=username).exists():
            return render(request, "register.html", {"error_message": "Username already taken"})

        user = User.objects.create_user(username=username, email=email, password=password)
        user.save()

        login(request, user)
        return redirect("check_medical")

    return render(request, "register.html")

def user_login(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("check_medical")
        else:
            return render(request, "login.html", {"error_message": "Invalid credentials"})

    return render(request, "login.html")

def user_logout(request):
    logout(request)
    return redirect("home")