mkdir -p ~/.streamlit/

cat > ~/.streamlit/config.toml << EOF
[server]
port = $PORT
enableCORS = false
headless = true
EOF