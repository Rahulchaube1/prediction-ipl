# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import os # For joining paths
import matplotlib.pyplot as plt
import seaborn as sns

# --- Try to import Kaggle Hub ---
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="IPL Match Winner Predictor", page_icon="🏏")

# --- Global Variables / Configuration ---
# Configuration for data loading
LOAD_FROM_KAGGLEHUB = False # Set to True to attempt Kaggle Hub first
KAGGLE_DATASET_SLUG = "mohammedplots/ipl-complete-dataset-2008-2024"
KAGGLE_TARGET_FILENAME = "IPL_Matches_2008_2024.csv"
LOCAL_CSV_FILE_PATH = 'matches.csv' # Assumes 'matches.csv' is in the same directory as this script

# --- Caching Functions for Performance ---

@st.cache_data # Cache the data loading and initial processing
def load_and_preprocess_data():
    """
    Loads data from Kaggle Hub or local CSV, preprocesses it,
    and fits label encoders.
    Returns:
        pd.DataFrame: Processed DataFrame for modeling.
        dict: Fitted label encoders.
        str: Message about the data source.
        pd.DataFrame: Original loaded DataFrame for EDA.
    """
    data = None
    use_dummy_data = False
    data_source_message = ""
    original_data_for_eda = None

    if LOAD_FROM_KAGGLEHUB and KAGGLEHUB_AVAILABLE:
        try:
            st.write(f"Attempting to download dataset from Kaggle Hub: {KAGGLE_DATASET_SLUG}...")
            with st.spinner(f"Downloading {KAGGLE_DATASET_SLUG}..."):
                download_path = kagglehub.dataset_download(KAGGLE_DATASET_SLUG)
            st.write(f"Dataset downloaded to: {download_path}")
            full_file_path = os.path.join(download_path, KAGGLE_TARGET_FILENAME)

            if os.path.exists(full_file_path):
                data_temp = pd.read_csv(full_file_path)
                data_source_message = f"Kaggle Hub ({KAGGLE_DATASET_SLUG}/{KAGGLE_TARGET_FILENAME})"
                data = data_temp
                original_data_for_eda = data.copy()
            else:
                st.warning(f"Target file '{KAGGLE_TARGET_FILENAME}' not found in Kaggle download. Files: {os.listdir(download_path)}")
                # Fall through to local CSV
                pass
        except Exception as e:
            st.warning(f"Error loading from Kaggle Hub: {e}. Falling back to local CSV.")
            # Fall through

    if data is None: # If Kaggle load failed or was skipped
        st.write(f"Attempting to load dataset from local CSV: {LOCAL_CSV_FILE_PATH}...")
        try:
            data_temp = pd.read_csv(LOCAL_CSV_FILE_PATH)
            data_source_message = f"Local CSV ('{LOCAL_CSV_FILE_PATH}')"
            data = data_temp
            original_data_for_eda = data.copy()
        except FileNotFoundError:
            st.error(f"Local CSV file '{LOCAL_CSV_FILE_PATH}' not found. Using dummy data.")
            use_dummy_data = True
            data_source_message = "Dummy Data (Local CSV not found)"
        except Exception as e:
            st.error(f"Error loading local CSV: {e}. Using dummy data.")
            use_dummy_data = True
            data_source_message = "Dummy Data (Error loading local CSV)"

    if use_dummy_data or data is None:
        st.warning("Using dummy dataset for demonstration.")
        dummy_data_dict = {
            'id': range(1, 51), # Increased dummy data size
            'city': ['Bangalore', 'Chandigarh', 'Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Hyderabad', 'Jaipur', 'Pune', 'Ahmedabad'] * 5,
            'date': pd.to_datetime(pd.date_range(start='2024-04-01', periods=50, freq='D')),
            'season': ['2024'] * 50,
            'team1': ['Royal Challengers Bangalore', 'Punjab Kings', 'Delhi Capitals', 'Mumbai Indians', 'Kolkata Knight Riders'] * 10,
            'team2': ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Delhi Capitals'] * 10,
            'venue': ['M Chinnaswamy Stadium', 'Punjab Cricket Association Stadium, Mohali', 'Feroz Shah Kotla', 'Wankhede Stadium', 'Eden Gardens'] * 10,
            'toss_winner': ['Royal Challengers Bangalore', 'Chennai Super Kings', 'Rajasthan Royals', 'Mumbai Indians', 'Delhi Capitals'] * 10,
            'toss_decision': ['field', 'bat', 'field', 'bat', 'field'] * 10,
            'winner': ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals', 'Mumbai Indians', 'Delhi Capitals', 'Punjab Kings', 'Sunrisers Hyderabad', 'Gujarat Titans', 'Lucknow Super Giants', 'Royal Challengers Bangalore'] * 5,
            'result': ['runs', 'wickets', 'runs', 'wickets', 'runs'] * 10,
            'result_margin': [10, 6, 15, 7, 20] * 10,
        }
        data = pd.DataFrame(dummy_data_dict)
        original_data_for_eda = data.copy()
        data_source_message = "Dummy Data"

    # --- Preprocessing ---
    data.columns = data.columns.str.lower().str.replace(' ', '_')
    required_cols = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'winner', 'city']
    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        st.error(f"Dataset missing required columns: {missing_cols}. Available: {list(data.columns)}. Cannot proceed with model training.")
        return None, None, data_source_message, original_data_for_eda # Return None if critical columns missing

    df_processed = data[required_cols].copy()
    df_processed.dropna(subset=['winner', 'city', 'venue'], inplace=True)

    if df_processed.empty:
        st.error("DataFrame is empty after cleaning (dropping NaNs). Cannot proceed.")
        return None, None, data_source_message, original_data_for_eda

    team_name_corrections = {
        'Delhi Daredevils': 'Delhi Capitals', 'Kings XI Punjab': 'Punjab Kings',
        'Rising Pune Supergiant': 'Rising Pune Supergiants', 'Deccan Chargers': 'Sunrisers Hyderabad',
    }
    for col in ['team1', 'team2', 'toss_winner', 'winner']:
        if col in df_processed.columns:
             df_processed[col] = df_processed[col].replace(team_name_corrections)

    # --- Fit Encoders ---
    encoders = {}
    all_teams_list = pd.concat([df_processed['team1'], df_processed['team2'], df_processed['winner'], df_processed['toss_winner']]).dropna().unique()
    all_venues_list = df_processed['venue'].dropna().unique()
    all_cities_list = df_processed['city'].dropna().unique()
    all_toss_decisions_list = df_processed['toss_decision'].dropna().unique()

    if not all_teams_list.size > 0 : return None, None, "Error: No unique teams after processing.", original_data_for_eda

    team_encoder = LabelEncoder().fit(all_teams_list)
    encoders['Team'] = team_encoder
    df_processed['team1_encoded'] = team_encoder.transform(df_processed['team1'])
    df_processed['team2_encoded'] = team_encoder.transform(df_processed['team2'])
    df_processed['toss_winner_encoded'] = team_encoder.transform(df_processed['toss_winner'])
    df_processed['winner_encoded'] = team_encoder.transform(df_processed['winner']) # Target

    if all_venues_list.size > 0:
        encoders['Venue'] = LabelEncoder().fit(all_venues_list)
        df_processed['venue_encoded'] = encoders['Venue'].transform(df_processed['venue'])
    else: # Handle empty venues if it occurs
        st.warning("No unique venues found. Venue feature will be problematic.")
        encoders['Venue'] = LabelEncoder().fit(np.array(['unknown_venue']))
        df_processed['venue_encoded'] = encoders['Venue'].transform(['unknown_venue'] * len(df_processed))


    if all_cities_list.size > 0:
        encoders['City'] = LabelEncoder().fit(all_cities_list)
        df_processed['city_encoded'] = encoders['City'].transform(df_processed['city'])
    else:
        st.warning("No unique cities found.")
        encoders['City'] = LabelEncoder().fit(np.array(['unknown_city']))
        df_processed['city_encoded'] = encoders['City'].transform(['unknown_city'] * len(df_processed))

    if all_toss_decisions_list.size > 0:
        encoders['TossDecision'] = LabelEncoder().fit(all_toss_decisions_list)
        df_processed['toss_decision_encoded'] = encoders['TossDecision'].transform(df_processed['toss_decision'])
    else:
        st.warning("No unique toss decisions found.")
        encoders['TossDecision'] = LabelEncoder().fit(np.array(['unknown_decision']))
        df_processed['toss_decision_encoded'] = encoders['TossDecision'].transform(['unknown_decision'] * len(df_processed))


    return df_processed, encoders, data_source_message, original_data_for_eda


@st.cache_resource # Cache the trained model
def train_model(_df_processed, _encoders):
    """
    Trains the Random Forest model.
    Args:
        _df_processed (pd.DataFrame): The processed DataFrame with encoded features.
        _encoders (dict): The fitted label encoders.
    Returns:
        RandomForestClassifier: The trained model.
        float: Accuracy of the model on the test set.
        pd.Series: Feature importances.
        list: Feature columns used for training.
    """
    if _df_processed is None or 'winner_encoded' not in _df_processed.columns:
        st.error("Cannot train model: Processed data is invalid or missing target.")
        return None, 0.0, None, []

    feature_columns = ['team1_encoded', 'team2_encoded', 'venue_encoded', 'city_encoded', 'toss_winner_encoded', 'toss_decision_encoded']
    
    # Ensure all feature columns are actually present after encoding
    actual_feature_columns = [col for col in feature_columns if col in _df_processed.columns]
    if len(actual_feature_columns) != len(feature_columns):
        st.error(f"Not all required feature columns are available for training. Missing: {set(feature_columns) - set(actual_feature_columns)}")
        return None, 0.0, None, []
        
    X = _df_processed[actual_feature_columns]
    y = _df_processed['winner_encoded']

    if X.empty or y.empty:
        st.error("Feature set (X) or target (y) is empty. Cannot train model.")
        return None, 0.0, None, actual_feature_columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)

    if X_train.empty:
        st.error("Training data is empty after split. Cannot train model (dataset might be too small).")
        return None, 0.0, None, actual_feature_columns

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    with st.spinner("Training model..."):
        model.fit(X_train, y_train)

    accuracy = 0.0
    if not X_test.empty:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

    importances = pd.Series(model.feature_importances_, index=actual_feature_columns).sort_values(ascending=False)
    return model, accuracy, importances, actual_feature_columns

# --- Load Data and Train Model ---
df_processed, encoders, data_source_message, original_data = load_and_preprocess_data()

model = None
model_accuracy = 0.0
feature_importances = None
training_feature_columns = []

if df_processed is not None and encoders is not None:
    model, model_accuracy, feature_importances, training_feature_columns = train_model(df_processed, encoders)
else:
    st.error("Data loading or preprocessing failed. Model cannot be trained.")

# --- UI Sections ---
st.title("🏏 IPL Match Winner Predictor & Analyzer")
st.caption(f"Using data from: {data_source_message}")

# --- Sidebar for Navigation/Options ---
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Choose a section:", ["Match Prediction", "Exploratory Data Analysis (EDA)", "Model Insights"])

if df_processed is None or model is None:
    st.error("Application cannot run fully as data loading/model training failed. Please check data source and configurations.")
    st.stop() # Stop execution if model/data isn't ready

# --- Main Content based on Sidebar Selection ---

if app_mode == "Match Prediction":
    st.header("🔮 Predict Match Winner")

    if not encoders or 'Team' not in encoders or not hasattr(encoders['Team'], 'classes_'):
        st.error("Team encoder not available. Prediction UI cannot be built.")
    else:
        all_teams_sorted = sorted(list(encoders['Team'].classes_))
        all_venues_sorted = sorted(list(encoders['Venue'].classes_)) if 'Venue' in encoders and hasattr(encoders['Venue'], 'classes_') else ["Unknown Venue"]
        all_cities_sorted = sorted(list(encoders['City'].classes_)) if 'City' in encoders and hasattr(encoders['City'], 'classes_') else ["Unknown City"]
        all_toss_decisions_sorted = sorted(list(encoders['TossDecision'].classes_)) if 'TossDecision' in encoders and hasattr(encoders['TossDecision'], 'classes_') else ["bat", "field"]


        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Select Team 1:", all_teams_sorted, index=0)
            venue = st.selectbox("Select Venue:", all_venues_sorted, index=0 if all_venues_sorted[0] != "Unknown Venue" else 0)
            toss_winner = st.selectbox("Toss Winner:", [team1, None] + [t for t in all_teams_sorted if t != team1 and t != None], index=0, format_func=lambda x: x if x is not None else "Select Toss Winner")


        with col2:
            team2 = st.selectbox("Select Team 2:", [t for t in all_teams_sorted if t != team1], index=min(1, len(all_teams_sorted)-2) if len(all_teams_sorted)>1 else 0)
            city = st.selectbox("Select City:", all_cities_sorted, index=0 if all_cities_sorted[0] != "Unknown City" else 0)
            toss_decision = st.selectbox("Toss Decision:", all_toss_decisions_sorted, index=0)
        
        if team1 == team2:
            st.warning("Team 1 and Team 2 cannot be the same. Please select different teams.")
        elif toss_winner is None:
            st.warning("Please select the Toss Winner.")
        elif st.button("Predict Winner", type="primary", use_container_width=True):
            if not training_feature_columns:
                 st.error("Model was not trained with feature columns information. Cannot predict.")
            else:
                with st.spinner("Predicting..."):
                    input_data = {}
                    unrecognized_inputs = []
                    team_main_encoder = encoders['Team']

                    try: input_data['team1_encoded'] = team_main_encoder.transform([team1])[0]
                    except: unrecognized_inputs.append(f"Team 1 '{team1}'")
                    try: input_data['team2_encoded'] = team_main_encoder.transform([team2])[0]
                    except: unrecognized_inputs.append(f"Team 2 '{team2}'")
                    try: input_data['venue_encoded'] = encoders['Venue'].transform([venue])[0]
                    except: unrecognized_inputs.append(f"Venue '{venue}'")
                    try: input_data['city_encoded'] = encoders['City'].transform([city])[0]
                    except: unrecognized_inputs.append(f"City '{city}'")
                    try: input_data['toss_winner_encoded'] = team_main_encoder.transform([toss_winner])[0]
                    except: unrecognized_inputs.append(f"Toss Winner '{toss_winner}'")
                    try: input_data['toss_decision_encoded'] = encoders['TossDecision'].transform([toss_decision])[0]
                    except: unrecognized_inputs.append(f"Toss Decision '{toss_decision}'")

                    if unrecognized_inputs:
                        st.error(f"Prediction failed. Unrecognized inputs: {', '.join(unrecognized_inputs)}")
                    else:
                        input_df = pd.DataFrame([input_data])
                        # Ensure column order matches training data
                        try:
                            input_df_ordered = input_df[training_feature_columns]
                        except KeyError as e:
                            st.error(f"Error aligning input for prediction. Missing columns: {e}. Expected: {training_feature_columns}")
                        else:
                            prediction_encoded = model.predict(input_df_ordered)[0]
                            prediction_proba = model.predict_proba(input_df_ordered)[0]
                            predicted_winner_name = team_main_encoder.inverse_transform([prediction_encoded])[0]

                            st.subheader(f"Predicted Winner: 🎉 {predicted_winner_name} 🎉")

                            # Probabilities
                            try:
                                team1_idx_in_classes = list(team_main_encoder.classes_).index(team1)
                                prob_team1 = prediction_proba[team1_idx_in_classes] * 100
                            except ValueError:
                                prob_team1 = "N/A (Team 1 not in model classes)"
                            
                            try:
                                team2_idx_in_classes = list(team_main_encoder.classes_).index(team2)
                                prob_team2 = prediction_proba[team2_idx_in_classes] * 100
                            except ValueError:
                                prob_team2 = "N/A (Team 2 not in model classes)"


                            st.write(f"**Probability of {team1} winning:** {prob_team1:.2f}%" if isinstance(prob_team1, float) else f"{prob_team1}")
                            st.write(f"**Probability of {team2} winning:** {prob_team2:.2f}%" if isinstance(prob_team2, float) else f"{prob_team2}")

                            # Display probabilities as a simple bar chart
                            if isinstance(prob_team1, float) and isinstance(prob_team2, float):
                                prob_df = pd.DataFrame({
                                    'Team': [team1, team2],
                                    'Winning Probability (%)': [prob_team1, prob_team2]
                                })
                                st.bar_chart(prob_df.set_index('Team'))


elif app_mode == "Exploratory Data Analysis (EDA)":
    st.header("📊 Exploratory Data Analysis")
    if original_data is not None:
        st.subheader("Dataset Preview")
        st.dataframe(original_data.head())

        st.subheader("Basic Statistics (Numerical Columns)")
        try:
            st.dataframe(original_data.describe(include=np.number))
        except Exception as e:
            st.warning(f"Could not generate numerical stats: {e}")
        
        st.subheader("Basic Statistics (Object Columns)")
        try:
            st.dataframe(original_data.describe(include='object'))
        except Exception as e:
            st.warning(f"Could not generate object stats: {e}")


        st.subheader("Matches Won by Each Team")
        if 'winner' in original_data.columns:
            wins_counts = original_data['winner'].value_counts()
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.barplot(y=wins_counts.index, x=wins_counts.values, ax=ax, palette="viridis")
            ax.set_title("Total Matches Won per Team")
            ax.set_xlabel("Number of Wins")
            ax.set_ylabel("Team")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("'winner' column not found in original data for EDA.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Toss Decision Distribution")
            if 'toss_decision' in original_data.columns:
                toss_decision_counts = original_data['toss_decision'].value_counts()
                fig, ax = plt.subplots()
                ax.pie(toss_decision_counts, labels=toss_decision_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
                ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
                ax.set_title("Toss Decisions")
                st.pyplot(fig)
            else:
                st.warning("'toss_decision' column not found.")
        
        with col2:
            st.subheader("Matches Played Per Season")
            if 'season' in original_data.columns:
                season_counts = original_data['season'].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(10, 5))
                season_counts.plot(kind='bar', ax=ax, color=sns.color_palette("coolwarm"))
                ax.set_title("Matches per Season")
                ax.set_xlabel("Season")
                ax.set_ylabel("Number of Matches")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("'season' column not found.")

        st.subheader("Most Frequent Venues")
        if 'venue' in original_data.columns:
            venue_counts = original_data['venue'].value_counts().nlargest(15)
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(y=venue_counts.index, x=venue_counts.values, ax=ax, palette="mako")
            ax.set_title("Top 15 Venues by Matches Hosted")
            ax.set_xlabel("Number of Matches")
            ax.set_ylabel("Venue")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("'venue' column not found.")

    else:
        st.warning("Original data not available for EDA (likely data loading failed).")


elif app_mode == "Model Insights":
    st.header("🧠 Model Insights")
    st.subheader(f"Model Accuracy on Test Set: {model_accuracy*100:.2f}%")

    if feature_importances is not None and not feature_importances.empty:
        st.subheader("Feature Importances")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=feature_importances.values, y=feature_importances.index, ax=ax, palette="rocket")
        ax.set_title("Feature Importances from Random Forest Model")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Features are encoded representations of team names, venue, city, and toss information.")
    else:
        st.warning("Feature importances are not available.")

    st.subheader("About the Model")
    st.markdown("""
        - **Model Used:** Random Forest Classifier
        - **Features:** Team 1, Team 2, Venue, City, Toss Winner, Toss Decision (all label encoded).
        - **Target Variable:** Winning Team.
        - This model predicts the winner of an IPL match based on historical data patterns.
        - Accuracy can vary depending on the dataset quality and the inherent unpredictability of sports.
    """)

st.sidebar.markdown("---")
st.sidebar.info("This app uses historical IPL data to predict match outcomes and provide insights. Remember that sports predictions are not guarantees!")

