import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from database import cibil_db

# --- INITIAL SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

models = {}
label_encoders = {}

# --- CORE FUNCTIONS ---
def load_ml_models():
    """Load all ML models and encoders from disk into memory."""
    global models, label_encoders
    try:
        model_files = {
            'eligibility': 'eligibility_model.pkl', 'product': 'product_model.pkl',
            'amount': 'amount_model.pkl', 'tenure': 'tenure_model.pkl', 'rate': 'rate_model.pkl'
        }
        
        for name, file in model_files.items():
            if os.path.exists(file):
                models[name] = joblib.load(file)
            else:
                logger.error(f"Model file {file} not found")
                raise FileNotFoundError(f"Model file {file} not found")
        
        if os.path.exists('label_encoders.pkl'):
            label_encoders = joblib.load('label_encoders.pkl')
        else:
            logger.error("Label encoders file not found")
            raise FileNotFoundError("Label encoders file not found")
        
        logger.info("All ML models and encoders loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading ML models: {e}")
        raise

def encode_categorical_features(request: 'LoanRequest') -> tuple:
    """Encode categorical features using the loaded label encoders."""
    try:
        # Create mapping between API field names and encoder keys
        field_mapping = {
            'gender': 'Gender',
            'marital_status': 'Marital Status',
            'property_type': 'Type of Property (Rented/Owned)',
            'education': 'Education level',
            'employment': 'Employment Status'
        }
        
        # Encode each categorical feature
        gender_enc = label_encoders[field_mapping['gender']].transform([request.gender])[0]
        marital_enc = label_encoders[field_mapping['marital_status']].transform([request.marital_status])[0]
        property_enc = label_encoders[field_mapping['property_type']].transform([request.property_type])[0]
        education_enc = label_encoders[field_mapping['education']].transform([request.education])[0]
        employment_enc = label_encoders[field_mapping['employment']].transform([request.employment])[0]
        
        return gender_enc, marital_enc, property_enc, education_enc, employment_enc
    
    except KeyError as e:
        logger.error(f"Label encoder not found for field: {e}")
        raise ValueError(f"Label encoder not found for field: {e}")
    except ValueError as e:
        logger.error(f"Invalid value for categorical field: {e}")
        raise ValueError(f"Invalid categorical input values: {e}")

def generate_loan_recommendation(request: 'LoanRequest') -> Dict[str, Any]:
    """Generate a full loan recommendation based on user profile."""
    try:
        # Get CIBIL score from database using CIBIL ID
        cibil_score = cibil_db.get_cibil_score(request.cibil_id)
        
        if cibil_score is None:
            raise ValueError(f"CIBIL ID {request.cibil_id} not found in database")
        
        # Encode categorical features
        gender_enc, marital_enc, property_enc, education_enc, employment_enc = encode_categorical_features(request)
        
        # Create base profile with database CIBIL score
        base_profile = np.array([[
            request.age, gender_enc, marital_enc, property_enc,
            education_enc, employment_enc, request.experience,
            request.salary, cibil_score  # Use score from database
        ]])

        # Step 1: Check eligibility
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
                'recommendations': [
                    "Improve CIBIL score above 650",
                    "Increase income stability",
                    "Consider co-applicant option",
                    "Build credit history"
                ]
            }

        # Step 2: Predict product type
        product_type_encoded = models['product'].predict(base_profile)[0]
        product_type = label_encoders['Product Type'].inverse_transform([int(product_type_encoded)])[0]

        # Step 3: Predict loan amount
        product_profile = np.column_stack([base_profile, [[product_type_encoded]]])
        loan_amount = models['amount'].predict(product_profile)[0]

        # Step 4: Predict tenure
        amount_profile = np.column_stack([product_profile, [[loan_amount]]])
        tenure_months = models['tenure'].predict(amount_profile)[0]

        # Step 5: Predict interest rate
        full_profile = np.column_stack([amount_profile, [[tenure_months]]])
        interest_rate = models['rate'].predict(full_profile)[0]

        # Calculate additional metrics
        monthly_emi = (loan_amount / tenure_months) if tenure_months > 0 else 0
        tenure_years = tenure_months / 12  # Convert to years

        # Risk assessment and recommendations
        if eligibility_prob >= 0.8 and cibil_score >= 750:
            risk_level = "Low Risk"
            recommendations = [
                "Excellent credit profile",
                "Consider higher loan amount",
                "Negotiate for better interest rates",
                "Fast-track approval possible"
            ]
        elif eligibility_prob >= 0.6 and cibil_score >= 650:
            risk_level = "Medium Risk"
            recommendations = [
                "Good credit profile",
                "Standard processing time",
                "Consider income documents verification",
                "Maintain current CIBIL score"
            ]
        else:
            risk_level = "Medium-High Risk"
            recommendations = [
                "Provide additional income proof",
                "Consider guarantor option",
                "Start with smaller loan amount",
                "Work on improving CIBIL score"
            ]

        return {
            'eligibility_status': 'Eligible',
            'recommended_product_type': product_type,
            'optimal_loan_amount': round(float(loan_amount), 2),
            'tenure_in_years': round(tenure_years, 1),  # Return in years with decimal
            'interest_rate': round(float(interest_rate), 2),
            'eligibility_probability': round(eligibility_prob * 100, 2),
            'monthly_emi': round(monthly_emi, 2),
            'recommendations': recommendations
        }
    
    except Exception as e:
        logger.error(f"Error generating recommendation: {str(e)}")
        raise

# --- APPLICATION LIFESPAN (STARTUP/SHUTDOWN) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Loading ML models and initializing database...")
    try:
        # Load ML models
        load_ml_models()
        
        # Initialize database and load CIBIL data if needed
        try:
            # Check if database has data
            test_score = cibil_db.get_cibil_score("227133320")  # Test with first ID from CSV
            if test_score is None:
                logger.info("Loading CIBIL data from CSV...")
                cibil_db.load_csv_to_database('cibil_database.csv')
        except Exception as e:
            logger.warning(f"Could not load CIBIL data: {e}")
        
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
    
    yield
    logger.info("Application shutdown.")

# --- FASTAPI APP INITIALIZATION ---
app = FastAPI(
    title="Loan Recommendation API",
    description="Backend service for predicting loan eligibility and terms using CIBIL ID lookup.",
    version="4.0.2",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- PYDANTIC MODELS (API DATA STRUCTURES) ---
class LoanRequest(BaseModel):
    age: int = Field(..., ge=18, le=80, description="Age in years (18-80)")
    gender: str = Field(..., description="Gender: Male/Female")
    property_type: str = Field(..., description="Property Type: Rented/Owned")
    marital_status: str = Field(..., description="Marital Status: Single/Married/Divorced")
    education: str = Field(..., description="Education level")
    employment: str = Field(..., description="Employment status")
    experience: int = Field(..., ge=0, description="Work experience in years")
    salary: int = Field(..., gt=0, description="Annual salary as an integer")
    cibil_id: str = Field(..., description="CIBIL ID for score lookup")

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        allowed = ['Male', 'Female']
        v_title = v.strip().title()
        if v_title not in allowed:
            raise ValueError(f'Gender must be one of {allowed}')
        return v_title

    @field_validator('education')
    @classmethod
    def validate_education(cls, v):
        allowed = ['Postgraduate', 'Graduate', '12th Pass', '10th Pass', 'Phd', 'Diploma']
        if v not in allowed:
            raise ValueError(f'Education must be one of {allowed}')
        return v

    @field_validator('employment')
    @classmethod
    def validate_employment(cls, v):
        allowed = ['Self-Employed', 'Salaried', 'Government', 'Retired', 'Student']
        if v not in allowed:
            raise ValueError(f'Employment must be one of {allowed}')
        return v

    @field_validator('property_type', 'marital_status')
    @classmethod
    def normalize_text_inputs(cls, v):
        return v.strip().title()

class LoanResponse(BaseModel):
    eligibility_status: str
    recommended_product_type: str
    optimal_loan_amount: float
    tenure_in_years: float
    interest_rate: float
    eligibility_probability: float
    monthly_emi: float
    recommendations: List[str]

# --- API ENDPOINTS ---
@app.get("/", tags=["Health"])
async def root():
    return {"message": "Loan Recommendation API is running.", "version": "4.0.2"}

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": len(models) == 5,
        "encoders_loaded": bool(label_encoders),
        "database_connected": True
    }

@app.post("/validate-cibil", tags=["Validation"])
async def validate_cibil_id(cibil_id: str):
    """Validate CIBIL ID and return score"""
    try:
        # Validate CIBIL ID exists
        is_valid = cibil_db.validate_cibil_id(cibil_id)
        
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"CIBIL ID {cibil_id} not found in database"
            )
        
        # Get CIBIL score
        cibil_score = cibil_db.get_cibil_score(cibil_id)
        
        return {
            "cibil_id": cibil_id,
            "cibil_score": cibil_score,
            "is_valid": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating CIBIL ID: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error validating CIBIL ID"
        )

@app.post("/predict", response_model=LoanResponse, tags=["Prediction"])
async def predict_loan(request: LoanRequest):
    """
    Generate loan recommendation based on customer profile
    
    Uses CIBIL ID to lookup score from database and provides:
    - Eligibility status
    - Recommended product type  
    - Optimal loan amount
    - Tenure in years (with decimals)
    - Interest rate
    - Risk assessment
    - Personalized recommendations
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
