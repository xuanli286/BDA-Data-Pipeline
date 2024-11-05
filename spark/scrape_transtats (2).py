import os
import time
from enum import Enum

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select


# Define an Enum for the months
class Month(Enum):
    JANUARY = "1"
    FEBRUARY = "2"
    MARCH = "3"
    APRIL = "4"
    MAY = "5"
    JUNE = "6"
    JULY = "7"
    AUGUST = "8"
    SEPTEMBER = "9"
    OCTOBER = "10"
    NOVEMBER = "11"
    DECEMBER = "12"


# Set up Chrome options (e.g., headless mode or download settings)
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--disable-gpu")  # Applicable to Windows OS
chrome_options.add_argument("--window-size=1920x1080")  # Set a window size
chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
chrome_options.add_argument(
    "--disable-dev-shm-usage"
)  # Overcome limited resource issues

download_directory = r"C:\Users\kengb\Downloads\transtats"  # Set download directory

# Add download preferences
prefs = {
    "download.default_directory": download_directory,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True,
}
chrome_options.add_experimental_option("prefs", prefs)


# Function to initialize WebDriver and load the page
def initialize_driver():
    print("Initializing driver...")
    driver = webdriver.Chrome(
        options=chrome_options,
    )
    driver.set_page_load_timeout(600)
    driver.get(
        "https://transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr"
    )
    driver.implicitly_wait(30)  # Set higher wait time for page to load
    return driver


# Function to select desired checkboxes
def select_checkboxes(driver):
    desired_checkbox_ids = [
        "YEAR",
        "MONTH",
        "DAY_OF_MONTH",
        "DAY_OF_WEEK",
        "FL_DATE",
        "TAIL_NUM",
        "ORIGIN_AIRPORT_ID",
        "ORIGIN",
        "DEST_AIRPORT_ID",
        "DEST",
        "DEP_TIME",
        "DEP_DELAY",
        "CRS_DEP_TIME",
        "ARR_TIME",
        "CRS_ARR_TIME",
        "OP_UNIQUE_CARRIER",
        "OP_CARRIER_FL_NUM",
        "TAIL_NUM",
        "ACTUAL_ELAPSED_TIME",
        "CRS_ELAPSED_TIME",
        "AIR_TIME",
        "ARR_DELAY",
        "DEP_DELAY",
        "ORIGIN",
        "DEST",
        "DISTANCE",
        "TAXI_IN",
        "TAXI_OUT",
        "CANCELLED",
        "CANCELLATION_CODE",
        "DIVERTED",
        "CARRIER_DELAY",
        "WEATHER_DELAY",
        "NAS_DELAY",
        "SECURITY_DELAY",
        "LATE_AIRCRAFT_DELAY",
    ]

    table = driver.find_element(By.ID, "myTable")
    checkboxes = table.find_elements(By.XPATH, './/input[@type="checkbox"]')

    for checkbox in checkboxes:
        checkbox_id = checkbox.get_attribute("id")
        if checkbox_id in desired_checkbox_ids:
            if not checkbox.is_selected():
                checkbox.click()
        else:
            if checkbox.is_selected():
                checkbox.click()


# Function to check if a file is downloaded
def is_file_downloaded(year, month_value):
    month_str = month_value.zfill(2)  # Ensure month is zero-padded
    file_name = f"data_{year}{month_str}.zip"
    return file_name in os.listdir(download_directory)


# Function to rename the downloaded file
def rename_downloaded_file(year, month_value):
    time.sleep(3)
    original_file = os.path.join(download_directory, "DL_SelectFields.zip")
    month_str = month_value.zfill(2)  # Ensure month is zero-padded
    new_file = os.path.join(download_directory, f"data_{year}{month_str}.zip")

    if os.path.exists(original_file):
        os.rename(original_file, new_file)
        print(f"Renamed file to {new_file}")
    else:
        print(f"File {original_file} not found for renaming.")


# Main function to scrape data starting from a specific year and month
def scrape_data(start_year: int, start_month: Month, stop_point: str):
    # Initialize the driver
    driver = initialize_driver()
    print("Driver initialized.")

    # Select the desired checkboxes
    select_checkboxes(driver)
    print("Checkboxes selected.")

    # Locate dropdown elements for Year and Month
    year_dropdown = Select(driver.find_element(By.ID, "cboYear"))
    period_dropdown = Select(driver.find_element(By.ID, "cboPeriod"))

    # Loop through each year and month for scraping
    for year_option in year_dropdown.options:
        year = year_option.get_attribute("value")
        if int(year) < start_year:
            continue  # Skip years earlier than the starting point

        year_dropdown.select_by_value(year)  # Select the year

        # Refetch the period dropdown to reflect new state after year selection
        period_dropdown = Select(driver.find_element(By.ID, "cboPeriod"))

        for period_option in period_dropdown.options:
            month_value = period_option.get_attribute("value")
            if int(year) == start_year and int(month_value) < int(start_month.value):
                continue  # Skip months earlier than the starting point

            month_name = period_option.text
            period_dropdown.select_by_value(month_value)  # Select the month

            if os.path.exists(
                os.path.join(download_directory, f"data_{stop_point}.zip")
            ):
                return

            # Check if the file has already been downloaded
            if is_file_downloaded(year, month_value):
                print(f"File for {year}, {month_name} already downloaded, skipping.")
                continue

            print(f"Downloading file for {year}, {month_name}")
            # Trigger the download
            download_button = driver.find_element(By.ID, "btnDownload")
            download_button.click()

            # Wait for the file to be downloaded (instead of time.sleep, continuously check for file)
            while not os.path.exists(
                os.path.join(download_directory, "DL_SelectFields.zip")
            ):
                time.sleep(1)  # Wait 1 second before checking again

            # Rename the file
            rename_downloaded_file(year, month_value)

            print(f"Downloaded and renamed file for {year}, {month_name}")

    # Close the browser when done
    driver.quit()


# Example usage:
while True:
    try:
        stop_point = "202212"
        if os.path.exists(os.path.join(download_directory, f"data_{stop_point}.zip")):
            print(f"File data_{stop_point}.zip already exists. Exiting...")
            break  # Exit the loop if the file exists

        scrape_data(start_year=2022, start_month=Month.JANUARY, stop_point=stop_point)

    except Exception as e:
        print(e.with_traceback(None))
        print("Error occurred. Restarting...")
        continue
