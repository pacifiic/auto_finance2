"""
네이버 뉴스 API 수집 스크립트
Naver News API Fetcher

네이버 개발자 센터에서 무료 API 키 발급 필요
https://developers.naver.com/

사용법:
    export NAVER_CLIENT_ID='your_client_id'
    export NAVER_CLIENT_SECRET='your_client_secret'
    python scripts/fetch_naver_news.py
"""

import os
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# ========================================
# 네이버 API 설정 (직접 입력하거나 환경변수 사용)
# ========================================
CLIENT_ID = os.environ.get('NAVER_CLIENT_ID', '')
CLIENT_SECRET = os.environ.get('NAVER_CLIENT_SECRET', '')

# 저장 경로
DATA_DIR = Path(__file__).parent.parent / 'data' / 'news_sns'

# 검색 키워드
KEYWORDS = [
    '비트코인',
    '이더리움', 
    '암호화폐',
    '코인 시세',
    '가상화폐',
    'BTC',
    'ETH',
    '바이낸스',
    '업비트',
]


def fetch_naver_news(keyword: str, display: int = 100) -> List[Dict]:
    """
    네이버 뉴스 검색 API 호출
    
    Args:
        keyword: 검색어
        display: 가져올 개수 (최대 100)
        
    Returns:
        뉴스 기사 리스트
    """
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        'X-Naver-Client-Id': CLIENT_ID,
        'X-Naver-Client-Secret': CLIENT_SECRET,
    }
    params = {
        'query': keyword,
        'display': display,
        'sort': 'date',  # sim: 정확도순, date: 최신순
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])
            
            articles = []
            for item in items:
                # HTML 태그 제거
                title = item.get('title', '').replace('<b>', '').replace('</b>', '')
                desc = item.get('description', '').replace('<b>', '').replace('</b>', '')
                
                articles.append({
                    'title': title,
                    'description': desc,
                    'url': item.get('link', ''),
                    'original_link': item.get('originallink', ''),
                    'published_at': item.get('pubDate', ''),
                    'source': 'naver',
                    'keyword': keyword,
                })
            
            return articles
        else:
            print(f"  Error: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        print(f"  Exception: {e}")
        return []


def analyze_sentiment(text: str) -> str:
    """간단한 감성 분석"""
    positive = ['상승', '급등', '돌파', '호재', '강세', '반등', '랠리', '매수', '불장']
    negative = ['하락', '급락', '폭락', '악재', '약세', '매도', '곰장', '위험', '우려']
    
    text_lower = text.lower()
    
    pos_count = sum(1 for w in positive if w in text)
    neg_count = sum(1 for w in negative if w in text)
    
    if pos_count > neg_count:
        return 'positive'
    elif neg_count > pos_count:
        return 'negative'
    else:
        return 'neutral'


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("  네이버 뉴스 API 수집")
    print("=" * 60)
    print(f"  시간: {datetime.now().isoformat()}")
    
    # API 키 확인
    if not CLIENT_ID or not CLIENT_SECRET:
        print("""
⚠️ 네이버 API 키가 설정되지 않았습니다!

[발급 방법]
1. https://developers.naver.com/ 접속
2. 애플리케이션 등록 (무료)
3. '검색' API 선택  
4. Client ID와 Secret 받기

[설정 방법 1] 환경변수:
    export NAVER_CLIENT_ID='발급받은_ID'
    export NAVER_CLIENT_SECRET='발급받은_SECRET'

[설정 방법 2] 이 파일 직접 수정:
    CLIENT_ID = '발급받은_ID'
    CLIENT_SECRET = '발급받은_SECRET'
""")
        return
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    all_articles = []
    
    # 키워드별 뉴스 수집
    for keyword in KEYWORDS:
        print(f"\n  검색: {keyword}")
        articles = fetch_naver_news(keyword, display=100)
        all_articles.extend(articles)
        print(f"    → {len(articles)}개 수집")
    
    if not all_articles:
        print("\n❌ 수집된 기사가 없습니다.")
        return
    
    # DataFrame 생성 및 중복 제거
    df = pd.DataFrame(all_articles)
    df = df.drop_duplicates(subset=['url'])
    
    # 감성 분석
    df['sentiment'] = df.apply(
        lambda row: analyze_sentiment(row['title'] + ' ' + row['description']), 
        axis=1
    )
    
    # 날짜 파싱
    df['published_at'] = pd.to_datetime(df['published_at'], format='mixed', errors='coerce')
    df = df.sort_values('published_at', ascending=False)
    
    # 저장
    filepath = DATA_DIR / 'naver_crypto_news.csv'
    df.to_csv(filepath, index=False)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("  수집 완료!")
    print("=" * 60)
    print(f"  총 기사 수: {len(df)}개")
    print(f"  저장 위치: {filepath}")
    
    # 감성 분석 결과
    print("\n  [감성 분석 결과]")
    for sentiment, count in df['sentiment'].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {sentiment}: {count}개 ({pct:.1f}%)")
    
    # 최신 뉴스 5개
    print("\n  [최신 뉴스 5개]")
    for _, row in df.head(5).iterrows():
        title = row['title'][:50]
        sent = row['sentiment']
        print(f"    [{sent}] {title}...")


if __name__ == '__main__':
    main()
