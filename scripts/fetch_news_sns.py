"""
News & SNS Data Fetcher
뉴스 및 SNS 데이터 수집 스크립트

코인 관련 뉴스 및 영향력 있는 인물들의 SNS를 수집합니다.
- CoinDesk (암호화폐 뉴스 1위 매체)
- Elon Musk X(Twitter)
- Trump Truth Social

주의: SNS 스크래핑은 각 플랫폼의 ToS를 준수해야 합니다.
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from bs4 import BeautifulSoup

# 저장 경로
DATA_DIR = Path(__file__).parent.parent / 'data' / 'news_sns'

# User-Agent
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}


# =====================================================
# CoinDesk 뉴스 수집
# =====================================================

def fetch_coindesk_news(pages: int = 10) -> List[Dict]:
    """
    CoinDesk에서 암호화폐 뉴스 수집
    
    Args:
        pages: 수집할 페이지 수
        
    Returns:
        뉴스 기사 리스트
    """
    print("\n[CoinDesk News]")
    print("-" * 50)
    
    all_articles = []
    base_url = "https://www.coindesk.com/livewire"
    
    try:
        # CoinDesk API 엔드포인트 사용
        api_url = "https://www.coindesk.com/pf/api/v3/content/fetch/articles-by-section"
        
        for page in range(1, pages + 1):
            params = {
                'd': json.dumps({
                    "sectionSlug": "markets",
                    "offset": (page - 1) * 20,
                    "size": 20,
                    "includeSections": ["markets", "business", "policy", "tech"],
                }),
                '_website': 'coindesk'
            }
            
            response = requests.get(api_url, params=params, headers=HEADERS, timeout=10)
            
            if response.status_code != 200:
                print(f"  Page {page}: Failed (status {response.status_code})")
                # 대체: RSS 피드 사용
                break
            
            data = response.json()
            articles = data.get('content_elements', [])
            
            for article in articles:
                all_articles.append({
                    'title': article.get('headlines', {}).get('basic', ''),
                    'description': article.get('description', {}).get('basic', ''),
                    'url': f"https://www.coindesk.com{article.get('canonical_url', '')}",
                    'published_at': article.get('first_publish_date', ''),
                    'source': 'coindesk',
                    'tags': [tag.get('text', '') for tag in article.get('taxonomy', {}).get('tags', [])],
                })
            
            print(f"  Page {page}: {len(articles)} articles")
            time.sleep(0.5)
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  Trying RSS feed instead...")
        all_articles = fetch_coindesk_rss()
    
    print(f"  Total: {len(all_articles)} articles")
    return all_articles


def fetch_coindesk_rss() -> List[Dict]:
    """CoinDesk RSS 피드에서 뉴스 수집 (대체 방법)"""
    import feedparser
    
    rss_url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
    
    try:
        feed = feedparser.parse(rss_url)
        articles = []
        
        for entry in feed.entries:
            articles.append({
                'title': entry.get('title', ''),
                'description': entry.get('summary', ''),
                'url': entry.get('link', ''),
                'published_at': entry.get('published', ''),
                'source': 'coindesk_rss',
                'tags': [tag.get('term', '') for tag in entry.get('tags', [])],
            })
        
        return articles
        
    except Exception as e:
        print(f"  RSS Error: {e}")
        return []


# =====================================================
# X (Twitter) 스크래핑 - Nitter 프록시 사용
# =====================================================

def fetch_x_posts(username: str, max_posts: int = 100) -> List[Dict]:
    """
    X(Twitter) 포스트 수집 (Nitter 프록시 사용)
    
    주의: X의 공식 API는 유료입니다. Nitter는 무료 프록시입니다.
    
    Args:
        username: X 사용자명 (예: 'elonmusk')
        max_posts: 최대 수집 포스트 수
        
    Returns:
        포스트 리스트
    """
    print(f"\n[X/Twitter - @{username}]")
    print("-" * 50)
    
    # Nitter 인스턴스 목록 (공개 인스턴스들)
    nitter_instances = [
        "https://nitter.poast.org",
        "https://nitter.privacydev.net", 
        "https://nitter.woodland.cafe",
    ]
    
    posts = []
    
    for instance in nitter_instances:
        try:
            url = f"{instance}/{username}"
            response = requests.get(url, headers=HEADERS, timeout=15)
            
            if response.status_code != 200:
                print(f"  {instance}: Failed ({response.status_code})")
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Nitter 형식으로 트윗 파싱
            tweets = soup.select('.timeline-item')
            
            for tweet in tweets[:max_posts]:
                # 트윗 내용
                content_elem = tweet.select_one('.tweet-content')
                content = content_elem.get_text(strip=True) if content_elem else ''
                
                # 시간
                time_elem = tweet.select_one('.tweet-date a')
                timestamp = time_elem.get('title', '') if time_elem else ''
                
                # 링크
                link_elem = tweet.select_one('.tweet-link')
                link = f"https://x.com{link_elem.get('href', '')}" if link_elem else ''
                
                if content:
                    posts.append({
                        'username': username,
                        'content': content,
                        'timestamp': timestamp,
                        'url': link,
                        'source': 'x',
                    })
            
            print(f"  {instance}: {len(posts)} posts fetched")
            break  # 성공하면 종료
            
        except Exception as e:
            print(f"  {instance}: Error - {e}")
            continue
    
    if not posts:
        print("  Warning: Could not fetch from any Nitter instance")
        print("  X/Twitter requires paid API access for direct scraping")
    
    return posts


# =====================================================
# Truth Social 스크래핑
# =====================================================

def fetch_truth_social_posts(username: str = 'realDonaldTrump', max_posts: int = 50) -> List[Dict]:
    """
    Truth Social 포스트 수집
    
    Args:
        username: Truth Social 사용자명
        max_posts: 최대 수집 포스트 수
        
    Returns:
        포스트 리스트
    """
    print(f"\n[Truth Social - @{username}]")
    print("-" * 50)
    
    posts = []
    
    try:
        # Truth Social API (공개 프로필용)
        api_url = f"https://truthsocial.com/api/v1/accounts/lookup?acct={username}"
        
        response = requests.get(api_url, headers=HEADERS, timeout=10)
        
        if response.status_code == 200:
            user_data = response.json()
            user_id = user_data.get('id')
            
            if user_id:
                # 사용자의 포스트 가져오기
                statuses_url = f"https://truthsocial.com/api/v1/accounts/{user_id}/statuses"
                params = {'limit': max_posts}
                
                statuses_response = requests.get(statuses_url, params=params, headers=HEADERS, timeout=10)
                
                if statuses_response.status_code == 200:
                    statuses = statuses_response.json()
                    
                    for status in statuses:
                        # HTML 태그 제거
                        content = BeautifulSoup(status.get('content', ''), 'html.parser').get_text()
                        
                        posts.append({
                            'username': username,
                            'content': content,
                            'timestamp': status.get('created_at', ''),
                            'url': status.get('url', ''),
                            'source': 'truth_social',
                            'reblogs': status.get('reblogs_count', 0),
                            'favorites': status.get('favourites_count', 0),
                        })
                    
                    print(f"  Fetched: {len(posts)} posts")
        else:
            print(f"  Failed: Status {response.status_code}")
            
    except Exception as e:
        print(f"  Error: {e}")
    
    return posts


# =====================================================
# 감성 분석 (간단 버전)
# =====================================================

def analyze_sentiment(text: str) -> Dict:
    """
    간단한 감성 분석
    
    실제로는 BERT, FinBERT 등 사용 권장
    """
    positive_words = ['bull', 'moon', 'pump', 'buy', 'long', 'bullish', 'up', 'gain', 'profit', 'win']
    negative_words = ['bear', 'dump', 'sell', 'short', 'bearish', 'down', 'loss', 'crash', 'fail']
    
    text_lower = text.lower()
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    total = pos_count + neg_count
    if total == 0:
        return {'sentiment': 'neutral', 'score': 0.0}
    
    score = (pos_count - neg_count) / total
    
    if score > 0.2:
        sentiment = 'positive'
    elif score < -0.2:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return {'sentiment': sentiment, 'score': score}


# =====================================================
# 메인 실행
# =====================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("News & SNS Data Fetcher")
    print("=" * 60)
    print(f"Time: {datetime.now().isoformat()}")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. CoinDesk 뉴스
    print("\n" + "=" * 60)
    print("1. Fetching CoinDesk News")
    print("=" * 60)
    
    try:
        import feedparser
    except ImportError:
        os.system("pip install feedparser -q")
        import feedparser
    
    news = fetch_coindesk_news(pages=5)
    if news:
        # 감성 분석 추가
        for article in news:
            sentiment = analyze_sentiment(article.get('title', '') + ' ' + article.get('description', ''))
            article.update(sentiment)
        
        df_news = pd.DataFrame(news)
        filepath = DATA_DIR / 'coindesk_news.csv'
        df_news.to_csv(filepath, index=False)
        print(f"\n  ✅ Saved: {filepath}")
    
    # 2. Elon Musk X/Twitter
    print("\n" + "=" * 60)
    print("2. Fetching Elon Musk X Posts")
    print("=" * 60)
    
    elon_posts = fetch_x_posts('elonmusk', max_posts=50)
    if elon_posts:
        for post in elon_posts:
            sentiment = analyze_sentiment(post.get('content', ''))
            post.update(sentiment)
        
        df_elon = pd.DataFrame(elon_posts)
        filepath = DATA_DIR / 'elon_musk_x.csv'
        df_elon.to_csv(filepath, index=False)
        print(f"\n  ✅ Saved: {filepath}")
    
    # 3. Trump Truth Social
    print("\n" + "=" * 60)
    print("3. Fetching Trump Truth Social Posts")
    print("=" * 60)
    
    trump_posts = fetch_truth_social_posts('realDonaldTrump', max_posts=50)
    if trump_posts:
        for post in trump_posts:
            sentiment = analyze_sentiment(post.get('content', ''))
            post.update(sentiment)
        
        df_trump = pd.DataFrame(trump_posts)
        filepath = DATA_DIR / 'trump_truth_social.csv'
        df_trump.to_csv(filepath, index=False)
        print(f"\n  ✅ Saved: {filepath}")
    
    # 4. 메타데이터 저장
    metadata = {
        'fetched_at': datetime.now().isoformat(),
        'sources': {
            'news': 'CoinDesk (coindesk.com)',
            'elon_musk': 'X/Twitter (@elonmusk)',
            'trump': 'Truth Social (@realDonaldTrump)',
        },
        'counts': {
            'news': len(news),
            'elon_posts': len(elon_posts),
            'trump_posts': len(trump_posts),
        },
    }
    
    metadata_path = DATA_DIR / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  CoinDesk News: {len(news)} articles")
    print(f"  Elon Musk X: {len(elon_posts)} posts")
    print(f"  Trump Truth Social: {len(trump_posts)} posts")
    print(f"\n  Data saved to: {DATA_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
