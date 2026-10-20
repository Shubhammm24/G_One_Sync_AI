import Constants from 'expo-constants';
import { Platform } from 'react-native';

// Dynamically extract host IP from Expo packager (works automatically for physical phones on Wi-Fi)
const bundlerHost = Constants.expoConfig?.hostUri?.split(':')[0];
const isInvalidHost =
  !bundlerHost ||
  bundlerHost.includes('ngrok') ||
  bundlerHost.includes('expo') ||
  bundlerHost.startsWith('169.254');

const DEFAULT_HOST = isInvalidHost
  ? (Platform.OS === 'android' && !Constants.isDevice ? '10.0.2.2' : '192.168.29.138')
  : bundlerHost;

export const API_BASE_URL = `http://${DEFAULT_HOST}:8001`;
export const WS_BASE_URL = `ws://${DEFAULT_HOST}:8002/ws/live`;


export const RISK_COLORS = {
  CRITICAL: '#EF4444', // Red-500
  HIGH: '#F97316',     // Orange-500
  MODERATE: '#EAB308', // Yellow-500
  LOW: '#10B981',      // Emerald-500
};

export const RISK_BG_COLORS = {
  CRITICAL: 'rgba(239, 68, 68, 0.15)',
  HIGH: 'rgba(249, 115, 22, 0.15)',
  MODERATE: 'rgba(234, 179, 8, 0.15)',
  LOW: 'rgba(16, 185, 129, 0.15)',
};

export const THEME = {
  bgDark: '#0B0F19',
  cardBg: '#131B2E',
  cardBorder: '#1E293B',
  surface: '#1E293B',
  primary: '#3B82F6',
  primaryDark: '#1D4ED8',
  textPrimary: '#F8FAFC',
  textSecondary: '#94A3B8',
  textMuted: '#64748B',
  accentTeal: '#06B6D4',
};

export const VITAL_BOUNDS = {
  heart_rate: { min: 60, max: 100, unit: 'BPM', label: 'Heart Rate' },
  respiratory_rate: { min: 12, max: 20, unit: '/min', label: 'Resp Rate' },
  spo2_pct: { min: 95, max: 100, unit: '%', label: 'SpO2' },
  temperature_c: { min: 36.5, max: 37.5, unit: '°C', label: 'Temp' },
  systolic_bp: { min: 90, max: 120, unit: 'mmHg', label: 'Systolic' },
  diastolic_bp: { min: 60, max: 80, unit: 'mmHg', label: 'Diastolic' },
};

export const LAB_BOUNDS = {
  wbc_count: { min: 4.5, max: 11.0, unit: 'K/uL', label: 'WBC' },
  lactate: { min: 0.5, max: 2.0, unit: 'mmol/L', label: 'Lactate' },
  creatinine: { min: 0.7, max: 1.3, unit: 'mg/dL', label: 'Creatinine' },
};
