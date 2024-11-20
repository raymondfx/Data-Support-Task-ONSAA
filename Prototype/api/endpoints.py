from fastapi import APIRouter, UploadFile, File
from ml_models.data_cleaner import DataCleaner
from ml_models.model_trainer import ModelTrainer

router = APIRouter()


@router.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):

    #Data upload and initial processing
    cleaner = DataCleaner()
    processed_data = cleaner.clean_data(await file.read())

    # Train or update ML model

    
    trainer = ModelTrainer()
    trainer.train(processed_data)

    return {"status": "Data processed successfully"}