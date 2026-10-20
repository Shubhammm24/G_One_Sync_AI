/**
 * G_One_Sync AI — Mobile Root Application
 * =========================================
 * Initializes push notification handlers, status bar, and root navigator.
 */

import React, { useEffect } from 'react';
import { StatusBar } from 'expo-status-bar';
import { Platform, SafeAreaView, StyleSheet, View } from 'react-native';
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
      <View style={styles.inner}>
        <AppNavigator />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: THEME.bgDark,
    ...(Platform.OS === 'web'
      ? {
          minHeight: '100vh' as any,
          width: '100vw' as any,
          display: 'flex' as any,
        }
      : {}),
  },
  inner: {
    flex: 1,
    ...(Platform.OS === 'web'
      ? {
          maxWidth: 480,
          width: '100%',
          alignSelf: 'center',
          minHeight: '100vh' as any,
        }
      : {}),
  },
});
