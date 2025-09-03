from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional, List
import joblib
import numpy as np
import logging
import os
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models and encoders
models = {}
label_encoders = {}

# Load ML Models
def load_ml_models():
    """Load all ML models and encoders from disk into memory."""
    global models, label_encoders
    try:
        model_files = {
            'eligibility': 'eligibility_model.pkl',
            'product': 'product_model.pkl',
            'amount': 'amount_model.pkl',
            'tenure': 'tenure_model.pkl',
            'rate': 'rate_model.pkl'
        }
        for name, file in model_files.items():
            if os.path.exists(file):
                models[name] = joblib.load(file)
                logger.info(f"Loaded model: {name}")
            else:
                logger.error(f"Model file {file} not found.")
                raise FileNotFoundError(f"Model file {file} not found")

        if os.path.exists('label_encoders.pkl'):
            label_encoders = joblib.load('label_encoders.pkl')
            logger.info(f"Loaded label encoders: {list(label_encoders.keys())}")
        else:
            logger.error("Label encoders file not found.")
            raise FileNotFoundError("Label encoders file not found")

        logger.info("All ML models and encoders loaded successfully.")

    except Exception as e:
        logger.error(f"Error loading ML models: {e}")
        raise

# Encoding categorical features
def encode_categorical_features(request: 'LoanRequest') -> tuple:
    """Encode categorical features using the loaded label encoders."""
    try:
        gender_enc = label_encoders['Gender'].transform([request.gender])[0]
        marital_enc = label_encoders['Marital Status'].transform([request.marital_status])[0]
        property_enc = label_encoders['Type of Property (Rented/Owned)'].transform([request.property_type])[0]
        education_enc = label_encoders['Education level'].transform([request.education])[0]
        employment_enc = label_encoders['Employment Status'].transform([request.employment])[0]
        return gender_enc, marital_enc, property_enc, education_enc, employment_enc
    except KeyError as e:
        logger.error(f"Categorical value not seen during training: {e}")
        raise ValueError(f"The value provided for {e} is not a valid category.")

# Generate loan recommendation
def generate_loan_recommendation(request: 'LoanRequest') -> Dict[str, Any]:
    """Generate a full loan recommendation based on user profile."""
    gender_enc, marital_enc, property_enc, education_enc, employment_enc = encode_categorical_features(request)
    
    base_profile = np.array([[  
        request.age, gender_enc, marital_enc, property_enc,
        education_enc, employment_enc, request.experience, 
        request.salary, request.cibil_score
    ]])

    eligibility = models['eligibility'].predict(base_profile)[0]
    eligibility_prob = models['eligibility'].predict_proba(base_profile)[0][1]

    if eligibility == 0:
        return {
            'eligibility_status': 'Not Eligible',
            'recommended_product_type': 'None',
            'optimal_loan_amount': 0,
            'tenure_in_years': 0,
            'interest_rate': 0,
            'eligibility_probability': round(eligibility_prob * 100, 2),
            'monthly_emi': 0,
            'recommendations': ["Your profile does not meet the eligibility criteria at this time."]
        }

    product_type_encoded = models['product'].predict(base_profile)[0]
    product_type = label_encoders['Product Type'].inverse_transform([int(product_type_encoded)])[0]

    product_profile = np.column_stack([base_profile, [[product_type_encoded]]])
    loan_amount = models['amount'].predict(product_profile)[0]

    amount_profile = np.column_stack([product_profile, [[loan_amount]]])
    tenure_months = models['tenure'].predict(amount_profile)[0]

    full_profile = np.column_stack([amount_profile, [[tenure_months]]])
    interest_rate = models['rate'].predict(full_profile)[0]

    monthly_emi = (loan_amount / tenure_months) if tenure_months > 0 else 0

    return {
        'eligibility_status': 'Eligible',
        'recommended_product_type': product_type,
        'optimal_loan_amount': round(float(loan_amount), 2),
        'tenure_in_years': round(tenure_months / 12, 1),
        'interest_rate': round(float(interest_rate), 2),
        'eligibility_probability': round(eligibility_prob * 100, 2),
        'monthly_emi': round(monthly_emi, 2),
        'recommendations': ["Based on your profile, this is the recommended loan structure."]
    }

# Application lifespan with startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Loading ML models and initializing database...")
    try:
        # Load ML models
        load_ml_models()
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
    
    yield
    logger.info("Application shutdown.")

# FastAPI app initialization
app = FastAPI(
    title="Loan Recommendation API",
    description="Backend service for predicting loan eligibility and terms using ML models.",
    version="4.0.3",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class LoanRequest(BaseModel):
    age: int = Field(..., ge=18, le=80, description="Age in years (18-80)")
    gender: str = Field(..., description="Gender: Male/Female")
    property_type: str = Field(..., description="Property Type: Rented/Owned")
    marital_status: str = Field(..., description="Marital Status: Single/Married/Divorced")
    education: str = Field(..., description="Education level")
    employment: str = Field(..., description="Employment status")
    experience: int = Field(..., ge=0, description="Work experience in years")
    salary: int = Field(..., gt=0, description="Annual salary as an integer")
    cibil_score: int = Field(..., ge=300, le=900, description="CIBIL score (300-900)")

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        allowed = ['Male', 'Female']
        v_title = v.strip().title() if isinstance(v, str) else v
        if v_title not in allowed:
            raise ValueError(f'Gender must be one of {allowed}')
        return v_title

    @field_validator('education')
    @classmethod
    def validate_education(cls, v):
        allowed = ['Postgraduate', 'Graduate', '12th Pass', '10th Pass', 'Phd', 'Diploma']
        v_title = v.strip().title() if isinstance(v, str) else v
        if v_title not in allowed:
            raise ValueError(f'Education must be one of {allowed}')
        return v_title

    @field_validator('employment')
    @classmethod
    def validate_employment(cls, v):
        allowed = ['Self-Employed', 'Salaried', 'Government', 'Retired', 'Student']
        v_title = v.strip().title() if isinstance(v, str) else v
        if v_title not in allowed:
            raise ValueError(f'Employment must be one of {allowed}')
        return v_title

    @field_validator('property_type', 'marital_status')
    @classmethod
    def normalize_text_inputs(cls, v):
        return v.strip().title() if isinstance(v, str) else v

class LoanResponse(BaseModel):
    eligibility_status: str
    recommended_product_type: str
    optimal_loan_amount: float
    tenure_in_years: float
    interest_rate: float
    eligibility_probability: float
    monthly_emi: float
    recommendations: List[str]

# API Endpoints
@app.get("/", tags=["Health"])
async def root():
    return {"message": "Loan Recommendation API is running.", "version": "4.0.3"}

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": len(models) == 5,
        "encoders_loaded": bool(label_encoders),
        "available_encoders": list(label_encoders.keys()) if label_encoders else []
    }

@app.post("/predict", response_model=LoanResponse, tags=["Prediction"])
async def predict_loan(request: LoanRequest):
    """
    Generate loan recommendation based on customer profile
    """
    try:
        recommendation = generate_loan_recommendation(request)
        return LoanResponse(**recommendation)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="An internal error occurred during prediction."
        )

@app.get("/categories", tags=["Info"])
async def get_categories():
    """Get all available categories for form dropdowns"""
    return {
        "gender": ["Male", "Female"],
        "property_type": ["Rented", "Owned"],
        "marital_status": ["Single", "Married", "Divorced"],
        "education": ["Postgraduate", "Graduate", "12th Pass", "10th Pass", "Phd", "Diploma"],
        "employment": ["Self-Employed", "Salaried", "Government", "Retired", "Student"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
