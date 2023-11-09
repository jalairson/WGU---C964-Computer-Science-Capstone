import subprocess
import sys


# Runs first to install dependencies
def install_required_libraries():
    required_libraries = ['scikit-learn', 'matplotlib', 'numpy', 'pandas', 'Pillow', 'imbalanced-learn']

    for library in required_libraries:
        subprocess.run(["pip", "install", library])

    pass


install_required_libraries()

import tkinter as tk
from tkinter import scrolledtext, messagebox
from PIL import Image, ImageTk
import pandas as pd
import matplotlib as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import random
import matplotlib.pyplot as plt

# Initial steps to parse and read the dataset
# Loading the CSV
file_path = 'csv/spam_or_not_spam.csv'
data = pd.read_csv(file_path)

# Splitting the data into features and labels
X = data['email']  # Features (email contents)
y = data['label']  # Labels (0 or 1)

# Handling missing values in the 'email' column
data['email'].fillna('', inplace=True)

# Initializing the Vectorizer and fitting the feature data
vectorizer = TfidfVectorizer()
X_processed = vectorizer.fit_transform(data['email'])

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)

# Applying SMOTE for oversampling the minority class (spam)
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initializing and training the Naive Bayes classifier
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_resampled, y_train_resampled)

# Predictions on the test set
predictions = naive_bayes.predict(X_test)


# Returns report of accuracy metrics
def accuracy_report():
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return f"Accuracy: {accuracy}\nEvaluation report:\n{report}"


# A function to center tkinter windows
def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    wx = (screen_width - width) // 2
    wy = (screen_height - height) // 2

    window.geometry(f"{width}x{height}+{wx}+{wy}")


# Calls accuracy report in Tkinter
def display_accuracy_report():
    accuracy_text = accuracy_report()
    report_window = tk.Tk()
    report_window.geometry("800x600")
    report_window.title("Accuracy Report")

    # Creating a Text widget to display the accuracy report
    report_text = scrolledtext.ScrolledText(report_window)
    report_text.insert(tk.INSERT, accuracy_text)
    report_text.pack(expand=True, fill='both')
    report_text.config(bg="black", fg="white")
    rw_w = 800
    rw_h = 600
    center_window(report_window, rw_w, rw_h)

    def close_accuracy_report():
        report_window.destroy()

    exit_button = tk.Button(report_window, text="Close", command=close_accuracy_report, fg='white', bg='#1b1d1f',
                            activebackground="black", activeforeground='white')
    exit_button.pack(side="right", anchor="se", padx=20, pady=20, ipadx=10, ipady=10)

    report_window.mainloop()


# This function call will take a random email from the test dataset and show its contents with the metrics of the classifier
# Run this cell multiple times to see the results for different test emails
def display_random_email(classifier, vectorizer, X_test, y_test, original_data):

    # Selects a random index from the test set
    random_index = random.randint(0, X_test.shape[0] - 1)

    # Retrieves the email content and label from the test set
    email_content = X_test[random_index]
    actual_label = y_test.iloc[random_index]

    # Retrieves original CSV feature to display training emails real content
    original_email_content = original_data['email'].iloc[random_index]

    # Reshapes the model email content if needed
    email_content = email_content.reshape(1, -1)

    # Inverse transform of the sparse matrix back to text
    inverse_transformed_email = vectorizer.inverse_transform(email_content)
    inverse_transformed_email_text = ' '.join(inverse_transformed_email[0])

    # Prediction using the classifier and the probability for both classes (0 and 1)
    probabilities = classifier.predict_proba(email_content)
    spam_probability = probabilities[0][1]

    # Classifying the email using the classifier
    predicted_class = classifier.predict(email_content)

    # Output
    email_info = (
        f"Original Email Content:\n{original_email_content}\n\n"
        f"Classifier Prediction: {'Spam' if predicted_class == 1 else 'Not Spam'}\n"
        f"Actual Label: {'Spam' if actual_label == 1 else 'Not Spam'}\n"
        f"Spam Probability: {spam_probability:.4f}\n"
    )

    # Tkinter window initialization for random email
    window = tk.Tk()
    window.title("Email Information")
    window.geometry("1024x768")

    text_widget = tk.Text(window, bg="black", fg="white")
    text_widget.pack(expand=True, fill='both')

    text_widget.delete(1.0, tk.END)
    text_widget.insert(tk.END, email_info)

    w_w = 1024
    w_h = 768
    center_window(window, w_w, w_h)

    def refresh_email():
        window.destroy()
        display_random_email(classifier, vectorizer, X_test, y_test, original_data)

    def close_random_email():
        window.destroy()

    exit_button = tk.Button(window, text="Close", command=close_random_email, fg='white', bg='#1b1d1f',
                               activebackground="black", activeforeground='white')
    exit_button.pack(side="right", anchor="se", padx=20, pady=20, ipadx=10, ipady=10)

    refresh_button = tk.Button(window, text="Refresh", command=refresh_email, fg='white', bg='#7e42f5',
                               activebackground="#27154a", activeforeground='white')
    refresh_button.pack(side="right", anchor="se", padx=20, pady=20, ipadx=10, ipady=10)

    window.mainloop()


# Generating the classification report
report = classification_report(y_test, predictions, output_dict=True)
class_report = pd.DataFrame(report).T

# Extracting metrics for spam and not spam classes
metrics_spam = class_report.loc['1'][:-1]
metrics_not_spam = class_report.loc['0'][:-1]


# Classifier accuracy metrics
def classifier_accuracy_chart():
    x = range(len(metrics_spam))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, metrics_spam, width, label='Spam', color='gray')
    rects2 = ax.bar([i + width for i in x], metrics_not_spam, width, label='Not Spam', color='#7e42f5')

    ax.set_ylabel('Scores')
    ax.set_title('Evaluation Metrics by Class')
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(class_report.columns[:-1])
    ax.legend()
    ax.set_facecolor('black')
    plt.show()

    pass


def data_dist_pie():
    # Showing the distribution of spam emails vs ham emails in the csv dataset

    spam_count = data[data['label'] == 1].shape[0]
    ham_count = data[data['label'] == 0].shape[0]

    labels = 'Spam', 'Ham'
    sizes = [spam_count, ham_count]
    colors = ['gray', '#7e42f5']

    fig = plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Training Data distribution of Emails (Spam vs. Ham)')
    plt.show()

    fig.set_facecolor('black')

    pass


def classified_dist_pie():
    # Showing the distribution of emails classified as spam vs ham by the naive bayes classifier

    results = pd.DataFrame({'Predictions': predictions, 'Actual Labels': y_test})

    spam_count = results[results['Predictions'] == 1].shape[0]
    ham_count = results[results['Predictions'] == 0].shape[0]

    labels = 'Spam', 'Ham'
    sizes = [spam_count, ham_count]
    colors = ['gray', '#7e42f5']

    fig = plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Naive Bayes Classification of emails (Spam vs. Ham)')
    plt.show()

    fig.set_facecolor('black')

    pass


# Function to execute k-fold cross validation
def execute_cross_validation():
    X = X_train_resampled.toarray()
    y = y_train_resampled.values

    num_splits = 5
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

    accuracy_scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        naive_bayes = MultinomialNB()
        naive_bayes.fit(X_train, y_train)
        predictions = naive_bayes.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        accuracy_scores.append(accuracy)

    mean_accuracy = np.mean(accuracy_scores)
    messagebox.showinfo("Cross Validation Result", f"Mean accuracy across {num_splits} folds: {mean_accuracy}")


# Rendering Tkinter GUI
def main_window():
    original_data = pd.read_csv('csv/spam_or_not_spam.csv')

    window = tk.Tk()
    window.title("Naive Bayes Spam Classifier")
    window.geometry("1024x768")
    window.attributes('-fullscreen', True)
    window.configure(bg="black")

    # A nice title/banner
    banner = tk.Label(window, text="Naive Bayes Spam Classifier", fg="white", bg="#27154a", font=("Arial", 16),
                      height=2, bd=4, relief='solid')
    banner.pack(fill="x")

    # Frame to hold the navigation buttons
    frame = tk.Frame(window)
    frame.pack(expand=False, fill="both")
    frame.place(in_=window, anchor="c", relx=.5, rely=.5)
    frame.configure(bg="black")

    # A cat image for funsies
    img = Image.open("image/cat.png")
    img = img.resize((100, 100))
    img = ImageTk.PhotoImage(img)
    label = tk.Label(frame, image=img)
    label.image = img
    label.pack(side="top", anchor="c", pady=30)

    # Exit function for exit button
    def exit_window():
        window.destroy()

    # Buttons for main program functions
    data_dist_button = tk.Button(frame, text="Dataset Spam/Ham Distribution Pie Chart", command=data_dist_pie,
                                 fg="white", bg="#1b1d1f", activebackground="black", activeforeground='white', width=40,
                                 height=2)
    data_dist_button.pack(expand=False, anchor="center", pady=5)

    classified_dist_button = tk.Button(frame, text="Bayes Spam/Ham Distribution Pie Chart", command=classified_dist_pie,
                                       fg="white", bg="#1b1d1f", activebackground="black", activeforeground='white',
                                       width=40, height=2)
    classified_dist_button.pack(expand=False, anchor="center", pady=5)

    classifier_accuracy_button = tk.Button(frame, text="Bayes Classifier Accuracy Bar Chart",
                                           command=classifier_accuracy_chart, fg="white", bg="#1b1d1f",
                                           activebackground="black", activeforeground='white', width=40, height=2)
    classifier_accuracy_button.pack(expand=False, anchor="center", pady=5)

    display_random_email_button = tk.Button(frame, text="Classify A Test Email",
                                            command=lambda: display_random_email(naive_bayes, vectorizer, X_test,
                                                                                 y_test, original_data), fg="white",
                                            bg="#7e42f5", activebackground="#27154a", activeforeground='white',
                                            width=40, height=2)
    display_random_email_button.pack(expand=False, anchor="center", pady=5)

    accuracy_report_button = tk.Button(frame, text="Classifier Accuracy Report", command=display_accuracy_report,
                                       fg="white", bg="#1b1d1f", activebackground="black", activeforeground='white',
                                       width=40, height=2)
    accuracy_report_button.pack(expand=False, anchor="center", pady=5)

    exit_button = tk.Button(window, text="Exit", command=exit_window, bg='#f77f43', activebackground="#452413",
                            activeforeground='white')
    exit_button.pack(side="bottom", anchor="se", padx=64, pady=64, ipadx=20, ipady=20)

    cross_validation_button = tk.Button(frame, text="Perform Cross Validation", command=execute_cross_validation,
                                        fg="white", bg="#1b1d1f", activebackground="black", activeforeground='white',
                                        width=40, height=2)
    cross_validation_button.pack(expand=False, anchor="center", pady=5)

    window.mainloop()


# Runs dependency check/installation before creating the main program window
if __name__ == "__main__":
    install_required_libraries()
    main_window()
