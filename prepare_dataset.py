import pandas as pd
import numpy as np
from faker import Faker
import random
import os

REAL_NEWS_SAMPLES = [
    ("Scientists Discover New Exoplanet in Habitable Zone", "A team of international astronomers announced the discovery of a potentially habitable exoplanet using advanced spectroscopy techniques. The planet orbits a star similar to our sun."),
    ("Global GDP Growth Slows to 2.5% in Latest Quarter", "International monetary fund reports slower economic growth due to trade tensions. However, employment remains stable in major economies."),
    ("New Climate Report Shows Rising Sea Levels", "UN climate scientists warn of accelerating sea level rise due to melting ice sheets. Coastal regions need improved adaptation strategies."),
    ("Medical Breakthrough in Cancer Treatment", "Researchers announce promising results from phase 3 trials of new immunotherapy. The treatment shows 70% effectiveness in specific cancer types."),
    ("Tech Company Announces Renewable Energy Initiative", "Major tech corporation commits to 100% renewable energy by 2025. Investment includes solar and wind projects across multiple countries."),
    ("University Publishes Groundbreaking Study on Quantum Computing", "Research team demonstrates new quantum computing advancement. Findings could accelerate development of quantum algorithms."),
    ("Government Allocates Funds for Infrastructure Projects", "Billion-dollar infrastructure bill passes legislature. Projects include bridges, roads, and public transportation systems."),
    ("Agricultural Innovation Improves Crop Yields", "New farming techniques increase crop productivity by 30%. Method focuses on sustainable soil management practices."),
    ("Health Organization Updates Vaccination Guidelines", "WHO releases updated vaccination schedule for children. New guidelines incorporate latest clinical research findings."),
    ("Environmental Protection Agency Announces New Standards", "EPA implements stricter emissions standards for industry. Changes expected to reduce air pollution significantly."),
]

FAKE_NEWS_SAMPLES = [
    ("Doctors HATE This One Weird Trick", "Scientists discovered a secret remedy that pharmaceutical companies don't want you to know about. Click here to learn the miracle cure!"),
    ("Celebrity Shocks World with Secret Confession", "A-list actor reveals shocking truth about Hollywood. Industry insiders say this could destroy everything we know!"),
    ("Government Hiding Evidence of UFOs Says Anonymous Source", "Whistleblower claims government has alien technology for decades. Military admits to classified UFO sightings!"),
    ("This Food Supplement Will Change Your Life", "Billionaire doctor discovers fountain of youth formula. 92% of people see dramatic results in just 7 days!"),
    ("Politicians Involved in Massive Cover-up", "Leaked emails prove high-level corruption at government. Media refuses to report on this explosive scandal!"),
    ("Miracle Cure Banned by Big Pharma", "Natural remedy cures all diseases but corporations suppress research. See why doctors don't want you to know!"),
    ("Tech Billionaire Reveals Mind-Control Technology", "Shocking discovery shows government surveillance is even worse. Devices can read human thoughts!"),
    ("This One Food Will Eliminate Belly Fat Overnight", "Nutritionists hate this simple diet secret. Lose 30 pounds in one week without exercise!"),
    ("Conspiracy: World Leaders Are Actually Aliens", "Proof emerges that elite politicians are extraterrestrial. Classified documents expose intergalactic government!"),
    ("This Vitamin Cures Everything But Is Illegal to Sell", "Big health organizations suppress miracle supplement. Former FDA chief exposes the truth!"),
]

def generate_dataset(num_samples=2000, output_file='fake_news_dataset.csv'):
    """
    Generate a balanced dataset of real and fake news
    
    Parameters:
    -----------
    num_samples : int
        Total number of samples to generate (will be split evenly)
    output_file : str
        Output CSV filename
    """
    fake = Faker()
    data = []
    
    samples_per_class = num_samples // 2

    print(f"Generating {num_samples} news samples...")
    print(f" Real news {samples_per_class}")
    print(f" Fake news: {samples_per_class}\n")

    # Generate real news
    for i in range(samples_per_class):
        title, text = random.choice(REAL_NEWS_SAMPLES)
        # Add variations
        title = f"{title} - {fake.word().title()}"
        text = f"{text} {fake.sentence(nb_words=15)}"

        data.append({
            'title': title,
            'text': text,
            'label': 0,  # 0 for real
            'date': fake.date_between(start_date='-1y'),
            'source': random.choice(['Reuters', 'AP News', 'BBC', 'Scientific American', 'Nature'])
        })

    # Generate fake news
    for i in range(samples_per_class):
        title, text = random.choice(FAKE_NEWS_SAMPLES)
        # Add Variations
        title = f"{title} {random.choice(['EXPOSED', '2024', 'SHOCKING'])}"
        text = f"{text} {fake.sentence(nb_words=15)}"

        data.append({
            'title': title,
            'text': text,
            'label': 1, # 1 for fake
            'date': fake.date_between(start_date='-1y'),
            'source': random.choice(['Viral News Daily', 'Truth Exposed', 'Undergorund News', 'Secret Sources'])
        })

    # Create Dataframe
    df = pd.DataFrame(data)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save
    df.to_csv(output_file, index=False)

    print(f"Dataset saved to '{output_file}'")
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Real news: {len(df[df['label'] == 0])}")
    print(f"  Fake news: {len(df[df['label'] == 1])}")
    print(f"  Columns: {', '.join(df.columns.tolist())}\n")

    return df

def load_and_analyze_dataset(filepath):
    """Load and analyze dataset"""

    print(f"\n Loading dataset from '{filepath}'...\n")

    df = pd.read_csv(filepath)

    print(f"Dataset Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nLabel Distribution: ")
    print(df['label'].value_counts())
    print(f"\nBasic Statistics: ")
    print(df.describe())
    print(f"\nMissing Values: ")
    print(df.isnull().sum())

    # Text Analysis
    df['title_length'] = df['title'].str.len()
    df['text_length'] = df['title'].str.len()
    df['title_words'] = df['title'].str.split().str.len()
    df['text_words'] = df['title'].str.split().str.len()

    print(f"\n Text Analysis:")
    print(f"Average Title Length: {df['title_length'].mean():.0f} characters")
    print(f"Average Text Length: {df['text_length'].mean():.0f} characters")
    print(f"Average Title Words: {df['title_words'].mean():.0f}")
    print(f"Average Text Words: {df['text_words'].mean():.0f}")

    return df

def download_public_datasets():
    """
    Information about publicly available fake news datasets
    """
    print("\n" + "="*70)
    print("PUBLIC FAKE NEWS DATASETS")
    print("="*70 + "\n")

    datasets = {
        "ISOT Fake News Dataset": {
            "url":"https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets",
            "source": "Kaggle",
            "size": "~44,000 articles",
            "description": "Collection of real and fake news from multiple souces",
            "features": ["title", "text", "label", "date"]
        },

        "News Credibility Dataset": {
            "url": "https://github.com/nehalmenon123-lang/Rumour-Identification",
            "source": "GitHub",
            "size": "~1,000+ roumors",
            "description": "Twitter rumor detection dataset",
            "features": ["tweet_text", "label", "user_info"]
        },

        "FakeNewsNet": {
            "url": "https://github.com/KaiDMML/FakeNewsNet",
            "source": "GitHub",
            "size": "~400k+ news articles",
            "description": "Fake news detection with newtwork information",
            "features": ["news_content", "social_context", "label"]
        },

        "Buzzfeed Fake News Dataset": {
            "url": "https://github.com/BuzzFeedNews/2018-07-wildfire-trends",
            "source": "BuzzFeed",
            "size": "~5,800 articles",
            "description": "2016 election misinformation dataset",
            "features": ["title", "text", "label"]
        },

        "Climate Change Dataset": {
            "url": "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset",
            "source": "Kaggle",
            "size": "~15,000 articles",
            "description": "Fake and real news about climate change",
            "features": ["title", "text", "label", "date"]
        }
    }
    
    for name, info in datasets.items():
        print(f" {name}")
        print(f" Source: {info['source']}")
        print(f" Size: {info['size']}")
        print(f" URL: {info['url']}")
        print(f" Description: {info['description']}")
        print(f" Features: {', '.join(info['features'])}")
        print()

def create_sample_dataset_csv():
    """Create a small sample dataset for testing"""
    
    sample_data = {
        'title': [
            'Trump Wins 2024 Election',
            'Scientists Discover New Species in Amazon',
            'Miracle Diet Pill Revealed',
            'COVID-19 Vaccine Approved by FDA',
            'Celebrities Unite for Charity',
            'Hidden Truth About Aliens Exposed',
            'New Cancer Treatment Shows Promise',
            'Government Cover-up Revealed',
            'Climate Summit Reaches Agreement',
            'This One Trick Will Shock You'
        ],
        'text': [
            'Donald Trump won the 2024 US Presidential Election with record turnout. Election results confirmed by election officials across all states.',
            'Brazilian researchers discovered three new species of frogs in the Amazon rainforest. The species were identified using DNA analysis and field observations.',
            'A new diet pill claims to burn fat without exercise. Scientists say this could be dangerous and lacks clinical evidence.',
            'The FDA approved a new COVID-19 vaccine after extensive clinical trials. The vaccine shows 95% efficacy against current variants.',
            'Hollywood celebrities organized a charity fundraiser raising $50 million for education. Event raised awareness for global literacy.',
            'Anonymous sources claim government has hidden evidence of extraterrestrial life. No official confirmation from any agency.',
            'Researchers at top medical institutions report positive results from cancer immunotherapy trials. Treatment shows promise for multiple cancer types.',
            'Leaked documents allegedly prove high-level government corruption. Media organizations are verifying the authenticity of documents.',
            'World leaders reached a historic agreement on climate action. New commitments aim to reduce carbon emissions by 50% by 2030.',
            'Health experts warn about side effects. One strange compound doctors dont want you to know about could change everything!'
        ],
        'label': [0, 0, 1, 0, 0, 1, 0, 1, 0, 1]
    }

    df = pd.DataFrame(sample_data)
    df.to_csv('sample_fake_news.csv', index=False)

    print("Sample dataset created: sample_fake_news.csv")
    print(df)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FAKE NEWS DATASET PREPARATION")
    print("="*70 + "\n")

    # Option 1: Generate synthetic dataset
    print("Option 1: Generating synthetic dataset...")
    generate_dataset(num_samples=2000, output_file='fake_news_dataset.csv')
    
    # Option 2: Analyze existing dataset
    if os.path.exists('fake_news_dataset.csv'):
        print("\nOption 2: Analyzing generated dataset...")
        load_and_analyze_dataset('fake_news_dataset.csv')
    
    # Option 3: Show public dataset sources
    download_public_datasets()
    
    # Option 4: Create sample dataset
    print("Creating sample dataset for quick testing...")
    create_sample_dataset_csv()
    
    print("\n" + "="*70)
    print("Dataset preparation complete!")
    print("="*70 + "\n")