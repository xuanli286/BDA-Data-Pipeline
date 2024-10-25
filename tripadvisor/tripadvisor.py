from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import time
# Save to a CSV file or another format
import pandas as pd
# Initialize WebDriver (Make sure to have the right path to chromedriver.exe)
# driver = webdriver.Chrome(executable_path='path/to/chromedriver.exe')
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-notifications")
chrome_options.add_argument("--disable-popup-blocking")
# chrome_options.add_argument("--start-maximized")

# Add user agent to avoid detection
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

# Initialize WebDriver
driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=chrome_options
)
wait = WebDriverWait(driver, 10)
# Load the TripAdvisor page (e.g., specific hotel or restaurant reviews page)
url = 'https://www.tripadvisor.com.sg/ShowUserReviews-g1-d8729020-r974457389-American_Airlines-World.html'
driver.get(url)

all_reviews = []

while True:
    try:
        # Wait for the reviews section to load
        WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, "//div[@class='review-container']")))

        # Locate all the reviews on the current page
        reviews_list = driver.find_elements(By.XPATH, "//div[@class='review-container']")

        # Loop through each review to extract data
        for review in reviews_list:
            try:
                # Extract the user ID (or other identifying information)
                userID = review.find_element(By.XPATH, ".//div[@class='username mo']").text.strip()
                print(userID)

                # Extract the review date (can be in multiple formats, so make sure it's handled correctly)
                publishedDate = review.find_element(By.XPATH, ".//span[contains(@class,'ratingDate')]").get_attribute('title')
                print(publishedDate)
                # Extract the review title
                title = review.find_element(By.XPATH, ".//span[@class='noQuotes']").text.strip()
                print(title)
                # Extract the review content
                review_text = review.find_element(By.XPATH, ".//p[@class='partial_entry']").text.strip()
                print(review_text)
                # Extract the rating (class name might change, adjust accordingly)
                rating = review.find_element(By.CLASS_NAME, "ui_bubble_rating").get_attribute("class")
                rating_value = rating.split("_")[-1][0]  # Extract the first digit as the rating (e.g., 4 from 'ui_bubble_rating bubble_40')
                print(rating_value)

                dateOfTravel = review.find_element(By.XPATH, ".//div[@class='prw_rup prw_reviews_stay_date_hsx']").text.strip()
                print(dateOfTravel)

                # Save or print extracted information
                review_data = {
                    'userID': userID,
                    'publishedDate': publishedDate,
                    'title': title,
                    'review': review_text,
                    'rating': rating_value,
                }
                print(review_data)
                all_reviews.append(review_data)

            except Exception as e:
                print(f"Error extracting review: {e}")
                continue

        # Click the "Next" button to go to the next page
        try:
            next_button = driver.find_element(By.CSS_SELECTOR, 'a.nav.next')
            if 'disabled' in next_button.get_attribute('class'):
                print("Reached the last page.")
                break
            next_button.click()  # Click the next button to load more reviews
            WebDriverWait(driver, 10).until(EC.staleness_of(reviews_list))  # Wait for the page to load
            time.sleep(3)  # Optional: allow some additional time to load
        except Exception as e:
            print(f"Could not find or click the 'Next' button: {e}")
            break

    except Exception as e:
        print(f"Error loading reviews page: {e}")
        break

# Close the WebDriver when done
driver.quit()



df = pd.DataFrame(all_reviews)
df.to_csv('tripadvisor_reviews.csv', index=False)

print(f'Total Reviews Scraped: {len(all_reviews)}')
