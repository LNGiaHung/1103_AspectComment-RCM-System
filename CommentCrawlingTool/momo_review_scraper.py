import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
from selenium.common.exceptions import StaleElementReferenceException
import re
import os

# --- CONFIG ---
URL = "https://www.momo.vn/cinema/review"
CINEMA_URL = "https://www.momo.vn/cinema"
TARGET_ADDITIONAL = 40  # need 40 more films

# --- SETUP SELENIUM ---
options = webdriver.ChromeOptions()
# Uncomment the next line to run without opening a browser window
# options.add_argument('--headless')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

driver = webdriver.Chrome(options=options)
driver.get(URL)
# Initial delay to let the page load fully
time.sleep(5)

wait = WebDriverWait(driver, 10)

def is_valid_review_link(href: str) -> bool:
    # Must be detail page under /cinema/<slug>-<id>/review with no query string
    if not href:
        return False
    if '?' in href or '#':
        # reject URLs with query or fragment
        if '?' in href or '#' in href:
            return False
    if href.rstrip('/') == "https://www.momo.vn/cinema/review":
        return False
    # pattern: /cinema/<slug>-<digits>/review
    pattern = r"^https?://www\.momo\.vn/cinema/.+?-\d+/review/?$"
    return re.match(pattern, href) is not None

def load_crawled_movie_urls(path: str) -> set:
    crawled = set()
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8", newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    href = (row.get("movie_url") or "").strip()
                    if is_valid_review_link(href):
                        crawled.add(href)
        except Exception:
            pass
    return crawled

SKIP_URLS = load_crawled_movie_urls("movie_comments.csv")
review_links = set()
TARGET_TOTAL = TARGET_ADDITIONAL

# Wait for grid to load
try:
    grid = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.grid")))
except Exception as e:
    print("Grid not found or page did not load:", e)
    driver.quit()
    exit(1)

def collect_links():
    count_before = len(review_links)
    # 1) Anchors with visible 'Xem thêm' span (explicit card CTA)
    elems1 = driver.find_elements(By.XPATH, "//a[span[text()='Xem thêm']]")
    # 2) Any anchor linking to a cinema review detail page
    elems2 = driver.find_elements(By.XPATH, "//a[contains(@href,'/cinema/') and contains(@href,'/review')]")
    for a in (elems1 + elems2):
        try:
            href = a.get_attribute("href")
        except StaleElementReferenceException:
            continue
    if href and is_valid_review_link(href) and href not in review_links and href not in SKIP_URLS:
            review_links.add(href)
    return len(review_links) - count_before
# Fallback collector from general cinema listing -> derive /review URLs and verify pages having review grid
def collect_from_cinema_listing():
    base_links = set()
    try:
        driver.get(CINEMA_URL)
        time.sleep(3)
    except Exception:
        return 0
    rounds = 0
    while rounds < 200:
        rounds += 1
        try:
            anchors = driver.find_elements(By.XPATH, "//a[contains(@href,'/cinema/') and not(contains(@href,'/review'))]")
        except Exception:
            anchors = []
        for a in anchors:
            try:
                href = a.get_attribute("href") or ""
            except StaleElementReferenceException:
                continue
            if not href:
                continue
            if '?' in href or '#' in href:
                continue
            # pattern: /cinema/<slug>-<digits>
            if re.match(r"^https?://www\.momo\.vn/cinema/.+?-\d+/?$", href):
                base_links.add(href.rstrip('/'))
        # scroll to load more
        try:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        except Exception:
            pass
        time.sleep(2.2)
        try:
            driver.execute_script("window.scrollBy(0, -200);")
        except Exception:
            pass
        time.sleep(0.8)
    # Verify candidates have review grid
    added = 0
    for base in list(base_links):
        review_url = base + "/review"
        if review_url in review_links or review_url in SKIP_URLS:
            continue
        if not re.match(r"^https?://www\.momo\.vn/cinema/.+?-\d+/review/?$", review_url):
            continue
        try:
            driver.get(review_url)
            time.sleep(2.5)
            wait = WebDriverWait(driver, 8)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.grid.grid-cols-1.divide-y.divide-gray-200")))
            review_links.add(review_url)
            added += 1
            if len(review_links) >= TARGET_TOTAL:
                break
        except Exception:
            continue
    return added

# Auto expand content: scroll and click potential load-more buttons until MIN_LINKS or no progress
no_progress_rounds = 0
max_rounds = 200
while len(review_links) < TARGET_TOTAL and no_progress_rounds < 30 and max_rounds > 0:
    max_rounds -= 1
    added = collect_links()
    # Try click any visible load more button
    try:
        buttons = driver.find_elements(By.XPATH, "//button[contains(., 'Xem thêm') or contains(., 'Xem tiếp') or contains(., 'Xem tiếp nhé')] | //a[contains(., 'Xem thêm')]")
        if buttons:
            try:
                driver.execute_script("arguments[0].scrollIntoView({behavior:'smooth', block:'center'});", buttons[-1])
            except Exception:
                pass
            time.sleep(0.8)
            try:
                buttons[-1].click()
                time.sleep(2.5)
            except Exception:
                pass
    except Exception:
        pass
    # Scroll to bottom to trigger lazy load
    try:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    except Exception:
        pass
    time.sleep(2.5)
    # Small scroll up to trigger intersection observers
    try:
        driver.execute_script("window.scrollBy(0, -200);")
    except Exception:
        pass
    # Additional small random scrolls
    try:
        driver.execute_script("window.scrollBy(0, 400);")
        driver.execute_script("window.scrollBy(0, -250);")
    except Exception:
        pass
    added += collect_links()
    if added == 0:
        no_progress_rounds += 1
    else:
        no_progress_rounds = 0

# Final collection pass
collect_links()
if len(review_links) < TARGET_TOTAL:
    collect_from_cinema_listing()

print(f"Found {len(review_links)} review links:")
for link in list(review_links)[:TARGET_TOTAL]:
    print(link)

print(f"Total review links collected: {len(review_links)}")

driver.quit()

# Save to text file
with open("review_links.txt", "w", encoding="utf-8") as f:
    for link in list(review_links)[:TARGET_TOTAL]:
        f.write(link + "\n")
print("Saved links to review_links.txt")

# Save to CSV file
with open("review_links.csv", "w", encoding="utf-8", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["link"])
    for link in list(review_links)[:TARGET_TOTAL]:
        writer.writerow([link])
print("Saved links to review_links.csv")
