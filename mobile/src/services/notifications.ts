/**
 * G_One_Sync AI — Mobile Push Notifications Service
 * ===================================================
 * Configures push notification listeners and registers device tokens.
 */

import * as Notifications from 'expo-notifications';
import { Platform } from 'react-native';
import { apiService } from './api';
import { getUsername } from '../utils/auth';

// Configure notification presentation rules (native only)
if (Platform.OS !== 'web') {
  Notifications.setNotificationHandler({
    handleNotification: async () => ({
      shouldShowAlert: true,
      shouldPlaySound: true,
      shouldSetBadge: true,
      shouldShowBanner: true,
      shouldShowList: true,
    }),
  });
}

export const registerForPushNotificationsAsync = async (): Promise<string | null> => {
  if (Platform.OS === 'web') {
    return null;
  }
  let token: string | null = null;

  try {
    const { status: existingStatus } = await Notifications.getPermissionsAsync();
    let finalStatus = existingStatus;

    if (existingStatus !== 'granted') {
      const { status } = await Notifications.requestPermissionsAsync();
      finalStatus = status;
    }

    if (finalStatus !== 'granted') {
      console.warn('Push notification permission not granted!');
      return null;
    }

    // Get device push token
    const tokenData = await Notifications.getDevicePushTokenAsync();
    token = tokenData.data;

    if (Platform.OS === 'android') {
      await Notifications.setNotificationChannelAsync('critical-alerts', {
        name: 'Critical Deterioration Alerts',
        importance: Notifications.AndroidImportance.MAX,
        vibrationPattern: [0, 250, 250, 250],
        lightColor: '#EF4444',
        sound: 'critical_alert.wav',
      });
    }

    // Register token with backend
    const physicianId = getUsername() || 'dr.sharma';
    const deviceType = Platform.OS === 'ios' ? 'ios' : 'android';
    if (token) {
      await apiService.registerDevice(token, deviceType, physicianId);
      console.log('FCM Device Token registered successfully:', token);
    }
  } catch (err) {
    console.warn('Failed to obtain device push token:', err);
  }

  return token;
};
