FROM apify/actor-python:3.12
COPY requirements.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
# bake the model into the image so runs do not download it
RUN python -c "from gliclass import GLiClassModel; from transformers import AutoTokenizer; GLiClassModel.from_pretrained('knowledgator/gliclass-small-v1.0'); AutoTokenizer.from_pretrained('knowledgator/gliclass-small-v1.0')"
COPY . ./
CMD ["python", "-m", "src"]
