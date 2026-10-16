# G_One_Sync AI — Mobile Companion App (Phase 8)

> **Real-Time Clinical Deterioration Companion App for Attending Physicians**
> Built with React Native, TypeScript, Expo, and FastAPI.

---

## 📱 Features

- **Physician JWT Authentication**: Secure credentials flow with biometric fallback (Face ID / Fingerprint)
- **Live ICU Watchlist**: Real-time patient risk classification, trend indicators (improving, stable, worsening), and color-coded risk badges (CRITICAL, HIGH, MODERATE, LOW)
- **Deep Clinical Detail**: High-frequency vitals grid with out-of-bounds highlighting, lab biomarker panels, and SHAP explainability waterfall
- **Emergency Action Buttons**: Immediate "Call ICU Nurse Station" and "Acknowledge Alert" actions
- **FCM Push Notifications**: Real-time push alerts for high/critical patient deterioration events

---

## 🛠️ Setup & Running

### Prerequisites
- Node.js ≥ 18
- npm or yarn
- Expo CLI (`npm install -g expo-cli`)

### Quickstart

```bash
# 1. Install dependencies
cd mobile
npm install

# 2. Start Expo development server
npm start

# 3. Run on Android emulator or physical device via Expo Go
npm run android

# 4. Run on iOS Simulator (macOS only)
npm run ios
```

---

## 🌐 API Configuration

By default, the mobile app targets the FastAPI serving API at:
- Android Emulator: `http://10.0.2.2:8001`
- iOS Simulator: `http://localhost:8001`

To change the endpoint, update `API_BASE_URL` in `src/utils/constants.ts` or set an environment variable.
