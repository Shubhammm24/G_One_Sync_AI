/**
 * G_One_Sync AI — Mobile Auth Helper & Biometric Service
 * ========================================================
 */

import * as LocalAuthentication from 'expo-local-authentication';
import { Platform } from 'react-native';

interface StoredSession {
  accessToken: string;
  refreshToken: string;
  username: string;
  expiresAt: number; // Unix timestamp in seconds
}

// In-memory session store (in native production this is backed by SecureStore / EncryptedSharedPreferences)
let currentSession: StoredSession | null = null;

export const saveSession = (
  username: string,
  accessToken: string,
  refreshToken: string,
  expiresInSeconds: number
): void => {
  currentSession = {
    username,
    accessToken,
    refreshToken,
    expiresAt: Math.floor(Date.now() / 1000) + expiresInSeconds,
  };
};

export const getAccessToken = (): string | null => {
  if (!currentSession) return null;
  // If expired, return null to force refresh
  if (Math.floor(Date.now() / 1000) >= currentSession.expiresAt) {
    return null;
  }
  return currentSession.accessToken;
};

export const getRefreshToken = (): string | null => {
  return currentSession?.refreshToken || null;
};

export const getUsername = (): string | null => {
  return currentSession?.username || null;
};

export const clearSession = (): void => {
  currentSession = null;
};

export const isSessionValid = (): boolean => {
  if (!currentSession) return false;
  return Math.floor(Date.now() / 1000) < currentSession.expiresAt;
};

/**
 * Authenticate using device biometrics (Face ID / Fingerprint)
 */
export const authenticateWithBiometrics = async (): Promise<boolean> => {
  if (Platform.OS === 'web') return false;
  try {
    const hasHardware = await LocalAuthentication.hasHardwareAsync();
    if (!hasHardware) return false;

    const isEnrolled = await LocalAuthentication.isEnrolledAsync();
    if (!isEnrolled) return false;

    const result = await LocalAuthentication.authenticateAsync({
      promptMessage: 'Authenticate to access ICU Patient Monitor',
      fallbackLabel: 'Use PIN/Password',
      cancelLabel: 'Cancel',
      disableDeviceFallback: false,
    });

    return result.success;
  } catch (error) {
    console.warn('Biometric authentication failed or unavailable:', error);
    return false;
  }
};
