# NLP-Project  
--

# **Amazon Product Reviews Scraper & Sentiment Analysis**  

## **Overview**  
This project is a **Flask-based web service** that scrapes product reviews from Amazon, performs **text preprocessing**, and conducts **sentiment analysis** using **Natural Language Processing (NLP) techniques**. It also extracts key insights using **TF-IDF vectorization** and **Doc2Vec embeddings**.  

## **Features**  
✅ Web scraping of Amazon product reviews  
✅ Sentiment analysis using **VADER**  
✅ Text cleaning & preprocessing (lemmatization, stopword removal, etc.)  
✅ **TF-IDF** & **Doc2Vec** for feature extraction  
✅ Flask API for easy integration  

## **Tech Stack**  
- **Python**  
- **Flask** (for API)  
- **BeautifulSoup** (for web scraping)  
- **NLTK** (for text processing)  
- **Gensim** (for Doc2Vec embeddings)  
- **Scikit-learn** (for feature extraction)  
- **Pandas** (for data handling)  

## **Installation & Setup**  
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/amazon-reviews-sentiment.git
   cd amazon-reviews-sentiment
   ```  
2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```  
3. **Run the Flask App**  
   ```bash
   python app.py
   ```  
4. **Access the API**  
   Open your browser or use Postman:  
   ```
   http://127.0.0.1:5000/scrape_reviews
   ```  

## **API Endpoints**  
| Endpoint          | Method | Description |
|------------------|--------|-------------|
| `/`              | GET    | Home route, returns service description |
| `/scrape_reviews` | GET    | Scrapes reviews from Amazon and returns sentiment analysis |  

## **Example Output**  
```json
{
   "Title": "Great laptop",
   "Ratings": 4.5,
   "Comments": "excellent performance very smooth",
   "Sentiment": "Positive"
}
```  

## **Next Steps**  
🚀 Deploying the API on **AWS/GCP**  
📊 Building an interactive dashboard for sentiment insights  
🔍 Expanding to scrape multiple e-commerce websites  

## **Contributions**  
Pull requests are welcome! For major changes, please open an issue first to discuss improvements.  

---
