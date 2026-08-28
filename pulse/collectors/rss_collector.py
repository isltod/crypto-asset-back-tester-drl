# pulse/collectors/rss_collector.py
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Any, Optional

class RSSCollector:
    """
    공식 크립토 언론사 RSS 피드를 주기적으로 파싱하여 신규 기사를 추출하는 수집기 (ARTICLE 타입)
    """
    def __init__(self, name: str, feed_url: str):
        self.name = name
        self.feed_url = feed_url

    def fetch_feed(self) -> List[Dict[str, Any]]:
        events = []
        try:
            req = urllib.request.Request(
                self.feed_url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_data = resp.read()
            
            root = ET.fromstring(xml_data)
            
            # 1. RSS 2.0 형식 (<rss><channel><item>)
            items = root.findall('.//item')
            if items:
                for item in items:
                    title = item.findtext('title') or ""
                    link = item.findtext('link') or ""
                    guid = item.findtext('guid') or link or title
                    pub_date = item.findtext('pubDate') or ""
                    description = item.findtext('description') or ""
                    
                    events.append({
                        'event_type': 'ARTICLE',
                        'source': self.name,
                        'guid': f"RSS_{self.name}_{guid.strip()}",
                        'title': title.strip(),
                        'link': link.strip(),
                        'summary': description.strip()[:1000],
                        'published_at': pub_date.strip()
                    })
                return events

            # 2. Atom 형식 (<feed><entry>)
            entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')
            if entries:
                for entry in entries:
                    title = entry.findtext('{http://www.w3.org/2005/Atom}title') or ""
                    link_elem = entry.find('{http://www.w3.org/2005/Atom}link')
                    link = link_elem.attrib.get('href', '') if link_elem is not None else ""
                    guid = entry.findtext('{http://www.w3.org/2005/Atom}id') or link or title
                    pub_date = entry.findtext('{http://www.w3.org/2005/Atom}published') or entry.findtext('{http://www.w3.org/2005/Atom}updated') or ""
                    summary = entry.findtext('{http://www.w3.org/2005/Atom}summary') or entry.findtext('{http://www.w3.org/2005/Atom}content') or ""
                    
                    events.append({
                        'event_type': 'ARTICLE',
                        'source': self.name,
                        'guid': f"RSS_{self.name}_{guid.strip()}",
                        'title': title.strip(),
                        'link': link.strip(),
                        'summary': summary.strip()[:1000],
                        'published_at': pub_date.strip()
                    })

        except Exception:
            pass

        return events
