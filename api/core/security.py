from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Add your authentication logic here
async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Implement your user authentication logic
    return {"username": "testuser"}