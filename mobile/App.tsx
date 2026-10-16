/**
 * G_One_Sync AI — Mobile Root Application
 * =========================================
 * Initializes push notification handlers, status bar, and root navigator.
 */

import React, { useEffect } from 'react';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaView, StyleSheet } from 'react-native';
import { AppNavigator } from './src/navigation/AppNavigator';
import { registerForPushNotificationsAsync } from './src/services/notifications';
import { THEME } from './src/utils/constants';

export default function App() {
  useEffect(() => {
    // Request FCM / push permissions on startup
    registerForPushNotificationsAsync();
  }, []);

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="light" />
      <AppNavigator />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: THEME.bgDark,
  },
});
