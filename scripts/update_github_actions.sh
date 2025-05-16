#!/bin/bash
# Update GitHub Actions workflow for improved SSH handling

set -e

WORKFLOW_FILE=".github/workflows/cd.yml"

echo "Updating GitHub Actions workflow file..."

# 1. Update the Deploy to Droplet step to use the newly generated key when available
sed -i '' '/- name: Deploy to Droplet/,/script: |/ s/key: \${{ secrets.SSH_PRIVATE_KEY }}/key: \${{ env.NEW_PRIVATE_KEY != '"''"' \&\& env.NEW_PRIVATE_KEY || secrets.SSH_PRIVATE_KEY }}/g' "$WORKFLOW_FILE"
sed -i '' '/- name: Deploy to Droplet/,/script: |/ s/use_insecure_cipher: true/use_insecure_cipher: \${{ env.SSH_USE_INSECURE == '"'true'"' }}/g' "$WORKFLOW_FILE"

# 2. Add a verification step at the end of the workflow
cat >> "$WORKFLOW_FILE" << 'END_OF_APPEND'

    - name: Verify Deployment
      run: |
        DROPLET_IP="${{ steps.terraform_apply.outputs.ip }}"
        echo "Verifying deployment on $DROPLET_IP..."
        
        # Wait for services to start
        echo "Waiting 30 seconds for services to start..."
        sleep 30
        
        # Check if application is responding
        echo "Checking if application is responding..."
        RETRY_COUNT=0
        MAX_RETRIES=5
        
        while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
          echo "Attempt $((RETRY_COUNT+1))/$MAX_RETRIES: Checking application status..."
          
          if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i ~/.ssh/do_key root@$DROPLET_IP 'docker ps | grep -q bank-nifty-trading'; then
            echo "✅ Application is running successfully!"
            
            # Save deployment info for reference
            echo "DEPLOY_SUCCESS=true" >> $GITHUB_ENV
            echo "DEPLOY_IP=$DROPLET_IP" >> $GITHUB_ENV
            echo "DEPLOY_TIME=$(date)" >> $GITHUB_ENV
            
            exit 0
          else
            echo "Application not fully running yet. Waiting..."
            sleep 20
            RETRY_COUNT=$((RETRY_COUNT+1))
          fi
        done
        
        echo "❌ Deployment verification failed. Check logs for details."
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i ~/.ssh/do_key root@$DROPLET_IP 'docker ps; docker logs $(docker ps -q --filter "name=bank-nifty")'
        echo "DEPLOY_SUCCESS=false" >> $GITHUB_ENV
END_OF_APPEND

echo "GitHub Actions workflow updated successfully!"
