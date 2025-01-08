# Package Lambda function
- name: Package Lambda function
  run: |
    cd lambda
    zip -r ../lambda-deployment-package.zip ./*
    cd ..
  shell: bash
