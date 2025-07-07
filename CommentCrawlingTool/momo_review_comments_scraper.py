import time
import csv
import random
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- CONFIG ---
INPUT_CSV = "review_links.csv"
OUTPUT_CSV = "movie_comments.csv"
MIN_COMMENTS = 10
MAX_COMMENTS = 50

options = webdriver.ChromeOptions()
# options.add_argument('--headless')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

def get_links_from_csv(filename):
    links = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            links.append(row['link'])
    return links

def save_comments_to_csv(comments, output_file):
    file_exists = os.path.isfile(output_file)
    with open(output_file, "a", encoding="utf-8", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["movie_url", "user", "date", "score", "comment"])
        if not file_exists:
            writer.writeheader()
        for row in comments:
            writer.writerow(row)

def slow_scroll_to_element(driver, element, steps=10, delay=0.2):
    """Scrolls slowly to the element in steps to ensure visibility."""
    driver.execute_script("window.scrollTo(0, 0);")
    location = driver.execute_script("return arguments[0].getBoundingClientRect().top + window.pageYOffset;", element)
    current = 0
    for i in range(steps):
        current = int(location * (i + 1) / steps)
        driver.execute_script(f"window.scrollTo(0, {current});")
        time.sleep(delay)
    driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", element)
    time.sleep(delay * 2)

def expand_all_xem_them_spans(driver):
    """Expand all 'Xem thêm' spans in currently loaded comments."""
    while True:
        expanded_any = False
        # Find all spans that match the selector and have 'Xem thêm' text
        spans = driver.find_elements(By.CSS_SELECTOR, "span.read-or-hide.cursor-pointer.pl-1.hover\:underline.text-blue-500")
        for span in spans:
            try:
                if 'Xem thêm' in span.text:
                    slow_scroll_to_element(driver, span, steps=10, delay=0.1)
                    span.click()
                    time.sleep(0.5)
                    expanded_any = True
            except Exception:
                continue
        if not expanded_any:
            break

def scrape_comments_for_movie(driver, url):
    driver.get(url)
    print("  Waiting 10s before crawling...")
    time.sleep(10)
    wait = WebDriverWait(driver, 10)
    comments = []

    # Expand comments until enough or no more button
    while True:
        try:
            grid = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.grid.grid-cols-1.divide-y.divide-gray-200")))
        except Exception as e:
            print(f"Grid not found for {url}: {e}")
            break

        # Count current comments
        comment_divs = driver.find_elements(By.CSS_SELECTOR, "div.grid.grid-cols-1.divide-y.divide-gray-200 > div.relative.w-full.py-5")
        if len(comment_divs) >= MAX_COMMENTS:
            break

        # Try to click the "Xem tiếp nhé!" button below the grid
        try:
            btn = driver.execute_script('''
                var grid = document.querySelector('div.grid.grid-cols-1.divide-y.divide-gray-200');
                if (!grid) return null;
                var el = grid.nextElementSibling;
                while (el) {
                    if (el.tagName === 'DIV' && el.classList.contains('mb-3') && el.classList.contains('mt-1') && el.classList.contains('text-center')) {
                        var btn = el.querySelector('button');
                        if (btn) return btn;
                    }
                    el = el.nextElementSibling;
                }
                return null;
            ''')
            if btn:
                slow_scroll_to_element(driver, btn, steps=20, delay=0.15)
                time.sleep(0.5)
                btn.click()
                delay = random.uniform(3, 6)
                print(f"  Waiting {delay:.1f}s after clicking 'Xem tiếp nhé!'...")
                time.sleep(delay)
            else:
                break
        except Exception as e:
            print(f"No more 'Xem tiếp nhé!' button or error: {e}")
            break

    # Expand all 'Xem thêm' spans before extracting comments
    expand_all_xem_them_spans(driver)

    # After loading, extract all comments
    comment_divs = driver.find_elements(By.CSS_SELECTOR, "div.grid.grid-cols-1.divide-y.divide-gray-200 > div.relative.w-full.py-5")
    for div in comment_divs[:MAX_COMMENTS]:
        try:
            user = div.find_element(By.CSS_SELECTOR, "div.text-md.text-gray-800").text
        except:
            user = ""
        try:
            date = div.find_element(By.CSS_SELECTOR, "div.text-xs.text-gray-500").text.strip()
        except:
            date = ""
        try:
            score = div.find_element(By.CSS_SELECTOR, "div.flex.items-center.text-md > span.pl-0\\.5").text.strip()
        except:
            score = ""
        try:
            comment = div.find_element(By.CSS_SELECTOR, "div.text-md.whitespace-pre-wrap.break-words.leading-relaxed.text-gray-900").text
        except:
            comment = ""
        comments.append({
            "movie_url": url,
            "user": user,
            "date": date,
            "score": score,
            "comment": comment
        })
    return comments

def main():
    links = get_links_from_csv(INPUT_CSV)
    driver = webdriver.Chrome(options=options)

    for idx, url in enumerate(links):
        print(f"Scraping {idx+1}/{len(links)}: {url}")
        comments = scrape_comments_for_movie(driver, url)
        print(f"  Found {len(comments)} comments")
        # Only save if at least MIN_COMMENTS
        if len(comments) >= MIN_COMMENTS:
            save_comments_to_csv(comments, OUTPUT_CSV)
        else:
            print(f"  Skipped (less than {MIN_COMMENTS} comments)")

    driver.quit()
    print(f"Saved all comments to {OUTPUT_CSV}")

if __name__ == "__main__":
    main() 