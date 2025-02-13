from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import sys

# if there are no CLI parameters
if len(sys.argv) <= 1:
    print('Ticker symbol CLI argument missing!')
    sys.exit(2)

# read the ticker from the CLI argument
ticker_symbol = sys.argv[1]

# build the URL of the target page
url = f'https://finance.yahoo.com/quote/{ticker_symbol}'

# Set up the Chrome options to enable headless mode
options = Options()
options.add_argument('--headless=new')  # Headless mode for Chrome

# Set up the service to manage the ChromeDriver
service = ChromeService(ChromeDriverManager().install())

# Initialize the Chrome WebDriver with options and service
driver = webdriver.Chrome(service=service, options=options)

# Now you can use the driver to navigate the page
driver.set_window_size(1920, 1080)
driver.get(url)

# Scraping logic...
print(f'Scraping data for {ticker_symbol} from: {url}')

# close the browser and free up the resources
driver.quit()
