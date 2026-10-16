/**
 * G_One_Sync AI — Mobile REST API Service
 * ==========================================
 * Typed client for interacting with the FastAPI serving backend.
 */

import {
  LoginResponse,
  PatientRiskSummary,
  WatchlistResponse,
} from '../types';
import { clearSession, getAccessToken, getRefreshToken, saveSession } from '../utils/auth';
import { API_BASE_URL } from '../utils/constants';

class ApiService {
  private baseUrl: string = API_BASE_URL;

  public setBaseUrl(url: string) {
    this.baseUrl = url;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      ...(options.headers as Record<string, string>),
    };

    const token = getAccessToken();
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    let response: Response;
    try {
      response = await fetch(url, {
        ...options,
        headers,
      });
    } catch (networkErr) {
      console.warn(`Network error reaching ${url}:`, networkErr);
      throw new Error(`Unable to reach server at ${this.baseUrl}. Check network or server status.`);
    }

    // Auto-refresh token on 401 if refresh token is available
    if (response.status === 401 && getRefreshToken() && endpoint !== '/auth/login') {
      const refreshed = await this.refreshToken();
      if (refreshed) {
        headers['Authorization'] = `Bearer ${getAccessToken()}`;
        response = await fetch(url, {
          ...options,
          headers,
        });
      }
    }

    if (!response.ok) {
      let errorDetail = `Request failed: HTTP ${response.status}`;
      try {
        const errorJson = await response.json();
        if (errorJson.detail) {
          errorDetail = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
        }
      } catch {
        // ignore json parse error
      }
      throw new Error(errorDetail);
    }

    return response.json();
  }

  /**
   * Physician login with username/password
   */
  async login(username: string, password: string): Promise<LoginResponse> {
    const data = await this.request<LoginResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });
    saveSession(username, data.access_token, data.refresh_token, data.expires_in);
    return data;
  }

  /**
   * Refresh JWT access token
   */
  async refreshToken(): Promise<boolean> {
    const refreshToken = getRefreshToken();
    if (!refreshToken) return false;

    try {
      const response = await fetch(`${this.baseUrl}/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });
      if (response.ok) {
        const data: LoginResponse = await response.json();
        saveSession('physician', data.access_token, data.refresh_token, data.expires_in);
        return true;
      }
    } catch (e) {
      console.warn('Failed to refresh token:', e);
    }

    clearSession();
    return false;
  }

  /**
   * Fetch current monitored patients watchlist
   */
  async getWatchlist(): Promise<WatchlistResponse> {
    return this.request<WatchlistResponse>('/mobile/watchlist', {
      method: 'GET',
    });
  }

  /**
   * Fetch detailed patient risk assessment and SHAP factors
   */
  async getPatientSummary(patientId: number): Promise<PatientRiskSummary> {
    return this.request<PatientRiskSummary>(`/mobile/patient/${patientId}`, {
      method: 'GET',
    });
  }

  /**
   * Register mobile device for FCM alerts
   */
  async registerDevice(
    deviceToken: string,
    deviceType: 'ios' | 'android',
    physicianId: string
  ): Promise<{ status: string }> {
    return this.request<{ status: string }>('/notifications/register', {
      method: 'POST',
      body: JSON.stringify({
        device_token: deviceToken,
        device_type: deviceType,
        physician_id: physicianId,
      }),
    });
  }

  /**
   * Trigger test push notification to verify push pipeline
   */
  async sendTestPush(deviceToken: string): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>(
      `/notifications/test?device_token=${encodeURIComponent(deviceToken)}`,
      { method: 'POST' }
    );
  }
}

export const apiService = new ApiService();
