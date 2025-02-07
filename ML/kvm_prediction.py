# Check for required packages
try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    import joblib
except ImportError as e:
    print("Error: Missing required packages. Please install them using:")
    print("pip3 install pandas scikit-learn joblib")
    exit(1)

# Load the dataset
df = pd.read_csv('../preped.csv')

# Select features and target
features = ['Hidden Gem Score', 'Runtime', 'Rotten Tomatoes Score', 
           'Metacritic Score', 'Awards Received', 'Awards Nominated For',
           'Boxoffice', 'IMDb Votes', 'Minimum Age'] + \
           [col for col in df.columns if col in ['Action', 'Adventure', 'Animation', 
           'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 
           'Fantasy', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 
           'News', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']]

target = 'IMDb Score'

# Preprocessing
X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train SVM model with RBF kernel
model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Save model and scaler
joblib.dump(model, 'kvm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Example prediction
example = X_test[0].reshape(1, -1)
prediction = model.predict(example)
print(f'Predicted IMDb Score: {prediction[0]:.2f}')