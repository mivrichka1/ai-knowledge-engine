import { defineConfig } from '@playwright/test';
export default defineConfig({testDir:'./e2e',timeout:180_000,retries:0,use:{baseURL:'http://127.0.0.1:5173',headless:true,launchOptions:process.env.PLAYWRIGHT_EXECUTABLE_PATH ? {executablePath:process.env.PLAYWRIGHT_EXECUTABLE_PATH} : {}}});
