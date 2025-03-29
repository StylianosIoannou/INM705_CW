# Create a virtual environment called "venv"
python -m venv venv

# Activate the virtual environment.
venv/Scripts/activate

# bypass policy for session
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
venv\Scripts\activate

# Upgrade pip to the latest version
pip install --upgrade pip

# Install all dependencies listed in Requirements.txt
pip install -r Requirements.txt