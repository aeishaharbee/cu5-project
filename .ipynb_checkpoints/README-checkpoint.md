# Hotel Price Prediction

Welcome to the **Hotel Price Prediction** repository! This project leverages machine learning to predict hotel prices based on various features and input parameters. The application is built with Python and features a user-friendly interface using **Streamlit**.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
The **Hotel Price Prediction** application aims to provide insights into hotel pricing trends and predict the cost of a hotel stay based on user-provided inputs. It uses machine learning models trained on historical data to deliver accurate predictions. This project is designed for users who want to explore and understand hotel pricing dynamics.

---

## Features
- User-friendly interface powered by **Streamlit**.
- Predict hotel prices based on input parameters such as location, star rating, accommodation type, and more.
- Visualization of predictions and data insights.
- Built with modular code for easy scalability and updates.

---

## Technologies Used
- **Python**: Core programming language.
- **Playwright & BeautifulSoup**: For scraping data from [Trivago](https://www.trivago.com.my/)
- **Streamlit**: For creating the interactive user interface.
- **Scikit-learn**: For building and training the machine learning model.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib/Seaborn**: For data visualization.

---

## Installation

### Prerequisites
- Python 3.7 or higher installed on your machine.
- Git installed (optional, for cloning the repository).

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/aeishaharbee/cu5-project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd cu5-project
   ```
4. Install the required dependencies:
   ```bash
   pip install streamlit pandas joblib plotly holidays datetime 
   ```

---

## Usage

1. Run the **Streamlit** application:
   ```bash
   streamlit run app.py
   ```

2. Open the provided local URL in your web browser (e.g., `http://localhost:8501`).

3. Input the required parameters in the web app to predict hotel prices and visualize the results.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- Special thanks to the contributors of **Streamlit**, **Scikit-learn**, and other libraries used in this project.

