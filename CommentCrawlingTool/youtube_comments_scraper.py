import argparse
import csv
import os
import random
import re
import sys
import time
from urllib.parse import urlparse, parse_qs

# Ensure selenium is available; auto-install if missing
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except ModuleNotFoundError:
    import subprocess
    try:
        print("selenium chưa được cài. Đang cài đặt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "selenium>=4.15.0"]) 
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        print("Đã cài selenium thành công.")
    except Exception as e:
        print(f"Không thể cài selenium tự động: {e}")
        raise

# Defaults for zero-arg automation mode
DEFAULT_INPUT_CSV = "youtube_video_links.csv"
DEFAULT_OUTPUT_CSV = "youtube_comments.csv"
DEFAULT_MAX_COMMENTS = 200
DEFAULT_HEADLESS = True
YT_MIN_TOKENS = 10  # inclusive lower bound
YT_MAX_TOKENS = 100  # inclusive upper bound


def normalize_youtube_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    # Convert shorts URL to watch URL
    if "/shorts/" in url:
        video_id = url.rstrip("/").split("/shorts/")[-1].split("?")[0]
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    # Ensure youtu.be short links become watch URLs
    parsed = urlparse(url)
    if parsed.netloc in {"youtu.be"}:
        video_id = parsed.path.lstrip("/")
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    return url


def ensure_utf8_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def build_driver(headless: bool) -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    if headless:
        # Use new headless where available; fall back if needed
        options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    options.add_argument("--lang=en-US,en;q=0.9,vi;q=0.8")
    # Stability flags (help fix sandbox/network service errors on Windows)
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    # Reduce automation signals
    options.add_experimental_option("excludeSwitches", ["enable-automation"]) 
    options.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(1280, 1000)
    return driver


def click_consent_if_present(driver: webdriver.Chrome):
    # Try a few common consent buttons
    possible_xpaths = [
        "//button//*[contains(text(), 'I agree') or contains(text(), 'Accept all')]/ancestor::button",
        "//button[contains(., 'I agree') or contains(., 'Accept all')]",
        "//button//*[contains(text(),'Tôi đồng ý')]/ancestor::button",
        "//button[contains(.,'Tôi đồng ý')]",
    ]
    for xp in possible_xpaths:
        try:
            btns = driver.find_elements(By.XPATH, xp)
            if btns:
                btns[0].click()
                time.sleep(1)
                break
        except Exception:
            continue


def scroll_to_comments_section(driver: webdriver.Chrome, wait: WebDriverWait):
    # Scroll to comments container
    for _ in range(10):
        driver.execute_script("window.scrollBy(0, 800);")
        time.sleep(0.5)
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#comments")))
            break
        except Exception:
            continue
    # Expand comments module if collapsed
    try:
        driver.execute_script(
            "const el = document.querySelector('#comments'); el && el.scrollIntoView({behavior: 'smooth', block: 'center'});"
        )
    except Exception:
        pass
    time.sleep(1)


def incremental_scroll(driver: webdriver.Chrome, total_scrolls: int = 50):
    last_height = driver.execute_script("return document.documentElement.scrollHeight;")
    stagnation = 0
    for i in range(total_scrolls):
        driver.execute_script("window.scrollBy(0, document.documentElement.scrollHeight);")
        time.sleep(random.uniform(0.6, 1.2))
        new_height = driver.execute_script("return document.documentElement.scrollHeight;")
        if new_height == last_height:
            stagnation += 1
            if stagnation >= 3:
                break
        else:
            stagnation = 0
        last_height = new_height


def extract_comments(driver: webdriver.Chrome, max_comments: int) -> list:
    # YouTube uses various renderers; try tolerant selectors
    comments_data = []
    seen = set()

    def count_words(text: str) -> int:
        return len(re.findall(r"\w+", text, flags=re.UNICODE))

    def meets_length_policy(text: str) -> bool:
        tokens = count_words(text)
        return (tokens >= YT_MIN_TOKENS) and (tokens <= YT_MAX_TOKENS)

    def get_elements():
        # Try both classic and new renderers
        elems = driver.find_elements(By.CSS_SELECTOR, "ytd-comment-thread-renderer")
        if not elems:
            elems = driver.find_elements(By.CSS_SELECTOR, "ytd-comment-view-model")
        return elems

    no_growth_rounds = 0
    while len(comments_data) < max_comments and no_growth_rounds < 5:
        items = get_elements()
        before = len(comments_data)
        for it in items:
            try:
                # Comment text
                text_el = None
                for sel in [
                    "yt-formatted-string#content-text",
                    "#content-text",
                    "#comment-content #content-text",
                ]:
                    try:
                        text_el = it.find_element(By.CSS_SELECTOR, sel)
                        if text_el and text_el.text.strip():
                            break
                    except Exception:
                        continue
                if not text_el:
                    continue
                comment_text = text_el.text.strip()
                if not comment_text:
                    continue

                # Filter by token length (between 10 and 100 inclusive)
                if not meets_length_policy(comment_text):
                    continue

                # Author
                author = ""
                for sel in ["a#author-text", "#author-text", "#header-author a"]:
                    try:
                        author_el = it.find_element(By.CSS_SELECTOR, sel)
                        if author_el.text.strip():
                            author = author_el.text.strip()
                            break
                    except Exception:
                        continue

                # Published time
                published = ""
                for sel in [
                    "a#published-time yt-formatted-string",
                    "a#published-time",
                    "#header-author time",
                ]:
                    try:
                        p_el = it.find_element(By.CSS_SELECTOR, sel)
                        if p_el.text.strip():
                            published = p_el.text.strip()
                            break
                    except Exception:
                        continue

                # Likes
                likes = ""
                for sel in ["span#vote-count-middle", "#vote-count-middle"]:
                    try:
                        l_el = it.find_element(By.CSS_SELECTOR, sel)
                        if l_el.text is not None:
                            likes = l_el.text.strip()
                            break
                    except Exception:
                        continue

                key = (author, published, comment_text)
                if key in seen:
                    continue
                seen.add(key)

                comments_data.append(
                    {
                        "author": author,
                        "date": published,
                        "likes": likes,
                        "comment": comment_text,
                    }
                )
                if len(comments_data) >= max_comments:
                    break
            except Exception:
                continue

        if len(comments_data) == before:
            # Try to scroll a bit more to load new items
            driver.execute_script("window.scrollBy(0, 800);")
            time.sleep(0.8)
            no_growth_rounds += 1
        else:
            no_growth_rounds = 0

    return comments_data


def save_to_csv(rows: list, output_csv: str, video_url: str):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video_url", "author", "date", "likes", "comment"]
        )
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "video_url": video_url,
                    "author": r.get("author", ""),
                    "date": r.get("date", ""),
                    "likes": r.get("likes", ""),
                    "comment": r.get("comment", ""),
                }
            )


def crawl_youtube_comments(video_url: str, max_comments: int, headless: bool, output_csv: str):
    ensure_utf8_stdout()
    url = normalize_youtube_url(video_url)
    driver = build_driver(headless)
    try:
        driver.get(url)
        time.sleep(3)
        wait = WebDriverWait(driver, 12)
        click_consent_if_present(driver)
        # Ensure player loaded a bit
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#player")))
        except Exception:
            pass
        scroll_to_comments_section(driver, wait)
        # Initial short wait for comments bootstrap
        time.sleep(2)
        incremental_scroll(driver, total_scrolls=80)
        comments = extract_comments(driver, max_comments)
        print(f"Collected {len(comments)} comments from: {url}")
        if comments:
            save_to_csv(comments, output_csv, url)
            print(f"Saved to {output_csv}")
        else:
            print("No comments collected.")
    finally:
        driver.quit()


def parse_args():
    p = argparse.ArgumentParser(description="YouTube comments scraper (Selenium)")
    # Not required; we support zero-arg automation mode
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--url", type=str, help="YouTube video URL")
    src.add_argument("--input_csv", type=str, help="CSV with a 'url' column")
    p.add_argument("--output_csv", type=str, default="youtube_comments.csv", help="Output CSV path")
    p.add_argument("--max_comments", type=int, default=200, help="Max comments per video")
    p.add_argument("--headless", action="store_true", help="Run Chrome in headless mode")
    args, _unknown = p.parse_known_args()
    return args


def main():
    args = parse_args()

    # Zero-argument automation mode (also triggers if only unknown args were passed)
    if not any([args.url, args.input_csv]):
        input_csv = DEFAULT_INPUT_CSV
        output_csv = DEFAULT_OUTPUT_CSV
        max_comments = DEFAULT_MAX_COMMENTS
        headless = DEFAULT_HEADLESS
        # Create template CSV if missing
        if not os.path.isfile(input_csv):
            with open(input_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["url"])
                # Add a placeholder row for user to fill in later
                w.writerow(["https://www.youtube.com/watch?v=XXXXXXXXXXX"])
            print(f"Created sample CSV: {input_csv}. Please add your video URLs and rerun.")
            return
        # Proceed to batch mode with defaults
        try:
            urls = []
            with open(input_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    u = (row.get("url") or row.get("link") or "").strip()
                    if u:
                        urls.append(u)
            if not urls:
                print(f"No URLs found in {input_csv}.")
                return
            for idx, u in enumerate(urls, start=1):
                print(f"[{idx}/{len(urls)}] {u}")
                try:
                    crawl_youtube_comments(u, max_comments, headless, output_csv)
                except Exception as e:
                    print(f"Failed: {u} -> {e}")
                    continue
        finally:
            return

    # CLI-driven modes
    if args.url:
        crawl_youtube_comments(args.url, args.max_comments, args.headless, args.output_csv)
        return

    if args.input_csv:
        urls = []
        with open(args.input_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                u = (row.get("url") or row.get("link") or "").strip()
                if u:
                    urls.append(u)
        for idx, u in enumerate(urls, start=1):
            print(f"[{idx}/{len(urls)}] {u}")
            try:
                crawl_youtube_comments(u, args.max_comments, args.headless, args.output_csv)
            except Exception as e:
                print(f"Failed: {u} -> {e}")
                continue
        return


if __name__ == "__main__":
    main()


