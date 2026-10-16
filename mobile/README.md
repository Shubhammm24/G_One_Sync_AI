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
# 1. Navigate to mobile directory
cd mobile

# 2. Run with Tunnel (Recommended for Physical Devices on any Wi-Fi/Hotspot)
# This bypasses Windows Firewall and router client isolation
npm run start:tunnel

# OR: Run on Local Wi-Fi (if phone is on the exact same Wi-Fi)
npm run start:lan

# OR: Run instantly in your Web Browser (no phone needed)
npm run web
```

Scan the QR code in **Expo Go** (available on Google Play Store & Apple App Store).

---

## 🌐 API Configuration

By default, the mobile app targets the FastAPI serving API at:
- Android Emulator: `http://10.0.2.2:8001`
- iOS Simulator: `http://localhost:8001`

To change the endpoint, update `API_BASE_URL` in `src/utils/constants.ts` or set an environment variable.
