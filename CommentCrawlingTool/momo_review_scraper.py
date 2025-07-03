import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv

# --- CONFIG ---
URL = "https://www.momo.vn/cinema/review"
MIN_LINKS = 50

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

input("Please manually expand the grid in the browser. When you are done, press Enter here to continue and collect the links...")

wait = WebDriverWait(driver, 10)

review_links = set()

# Wait for grid to load (in case user expanded more content)
try:
    grid = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.grid")))
except Exception as e:
    print("Grid not found or page did not load:", e)
    driver.quit()
    exit(1)

# Find all 'Xem thêm' links using Selenium only
links = driver.find_elements(By.XPATH, "//a[span[text()='Xem thêm']]")
for link in links:
    href = link.get_attribute("href")
    if href and href not in review_links:
        review_links.add(href)

print(f"Found {len(review_links)} review links:")
for link in list(review_links)[:MIN_LINKS]:
    print(link)

print(f"Total review links collected: {len(review_links)}")

driver.quit()

# Save to text file
with open("review_links.txt", "w", encoding="utf-8") as f:
    for link in list(review_links)[:MIN_LINKS]:
        f.write(link + "\n")
print("Saved links to review_links.txt")

# Save to CSV file
with open("review_links.csv", "w", encoding="utf-8", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["link"])
    for link in list(review_links)[:MIN_LINKS]:
        writer.writerow([link])
print("Saved links to review_links.csv")
