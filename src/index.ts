import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { GoogleAuth } from 'google-auth-library';
import { VertexAI } from '@google-cloud/vertexai';

dotenv.config();

const app = express();
const port = process.env.PORT || 5001;

app.use(cors());
app.use(express.json());

const auth = new GoogleAuth();
const projectId = (process.env.GCP_PROJECT_ID || process.env.GOOGLE_CLOUD_PROJECT)?.trim();
const region = (process.env.REGION || 'us-central1').trim();

// Initialize Vertex AI SDK client
const vertexAI = new VertexAI({ project: projectId, location: region });

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'healthy' });
});

/**
 * POST /process
 * Triggered by Google Cloud Tasks.
 * Orchestrates heavy AI transcription and summarization of a file.
 * Securely returns results back to the Web Backend's callbackUrl using Google OIDC.
 */
app.post('/process', async (req, res) => {
  const { fileId, userId, bucket, storagePath, callbackUrl, fileType } = req.body;

  console.log(`[AI Worker] Received processing request from Cloud Tasks at: ${new Date().toISOString()}`);
  console.log('[AI Worker] Payload parameters:');
  console.log(`  - fileId: ${fileId}`);
  console.log(`  - userId: ${userId}`);
  console.log(`  - bucket: ${bucket}`);
  console.log(`  - storagePath: ${storagePath}`);
  console.log(`  - callbackUrl: ${callbackUrl}`);
  console.log(`  - fileType: ${fileType || 'Auto-detect'}`);

  if (!fileId || !userId || !bucket || !storagePath || !callbackUrl) {
    console.error('[AI Worker] Validation Failed: Missing one or more required parameters.');
    return res.status(400).json({ error: 'Bad Request: Missing required parameters' });
  }

  console.log(`[AI Worker] Starting AI processing for file: ${fileId} (User: ${userId})`);

  try {
    // 1. Generate AI results using Vertex AI Gemini
    const fileTypeLower = (fileType || '').toLowerCase();
    const isPdf = fileTypeLower === 'pdf' || (!fileType && storagePath.toLowerCase().endsWith('.pdf'));
    const detectedType = isPdf ? 'PDF' : 'Video/Audio';
    console.log(`[AI Worker] Detected file type: ${detectedType} (from storage path and type hint)`);

    let summary = '';
    let transcription = '';

    // If running in local development mode without GCP credentials, fall back to mock AI data
    if (!projectId || process.env.NODE_ENV === 'development') {
      console.log('[AI Worker] Running in development mode or GCP_PROJECT_ID is missing. Simulating AI processing...');
      console.log('[AI Worker] Simulating processing delay (10 seconds)...');
      await new Promise((resolve) => setTimeout(resolve, 10000));

      if (isPdf) {
        summary = `### Executive Summary (Simulated)\n\nThis PDF document **"${fileId}"** was analyzed successfully using simulated Gemini intelligence.\n\n#### Key Findings:\n* **Cloud Architecture**: Decoupling web servers from compute workers avoids connection terminations.\n* **Server-Sent Events**: Delivers instant, multiplexed pushes directly to browsers under HTTP/2.\n* **State Integrity**: Client states align automatically upon webhook database commits.`;
        transcription = 'N/A (PDF Document File)';
      } else {
        summary = `### Video Summary (Simulated)\n\nThis video was transcribed and analyzed successfully. Key discussion points:\n* **Decoupled Workers**: Offloading Vertex AI requests ensures low-latency REST APIs for web users.\n* **Cloud Tasks**: Provides rate-limiting (e.g. 5 concurrent dispatches) to protect backend resource constraints.\n* **Heartbeat Pings**: Keeps connection sockets open across proxies under Google Cloud's GFE.`;
        transcription = `[00:01] Hello and welcome to OmniBrief AI.\n[00:05] Today we're configuring Server-Sent Events with Cloud Tasks.\n[00:10] The worker does the heavy processing and sends a callback to the backend when done.`;
      }
      console.log('[AI Worker] Simulated processing complete.');
    } else {
      // Clean up storage path to avoid malformed double slashes in the URI
      const cleanPath = storagePath.startsWith('/') ? storagePath.substring(1) : storagePath;
      const fileUri = `gs://${bucket}/${cleanPath}`;
      const mimeType = isPdf ? 'application/pdf' : 'video/mp4';

      console.log(`[AI Worker] Invoking Vertex AI Gemini 1.5 Flash for file: ${fileUri} (Mime Type: ${mimeType})`);

      const generativeModel = vertexAI.getGenerativeModel({
        model: 'gemini-1.5-flash',
      });

      const filePart = {
        fileData: {
          fileUri: fileUri,
          mimeType: mimeType,
        },
      };

      const promptText = isPdf
        ? "Analyze this PDF document. Provide a comprehensive summary in clear Markdown formatting, explaining the key points, main conclusions, and highlights."
        : "Analyze this video file. Provide two sections in your response:\n1. Summary: A detailed summary of the video content.\n2. Transcript: A chronological transcription of the speech in the video with timestamps.\nFormat your response in Markdown.";

      console.log(`[AI Worker] Sending prompt to Gemini: "${promptText}"`);
      const promptPart = { text: promptText };

      const request = {
        contents: [{ role: 'user', parts: [filePart, promptPart] }],
      };

      const startTime = Date.now();
      const responseResult = await generativeModel.generateContent(request);
      const response = await responseResult.response;
      console.log(`[AI Worker] Gemini generation API completed in ${((Date.now() - startTime) / 1000).toFixed(2)}s.`);

      const responseText = response.candidates?.[0]?.content?.parts?.[0]?.text;
      if (!responseText) {
        throw new Error('Empty response received from Gemini');
      }

      console.log(`[AI Worker] Received Gemini response text of length: ${responseText.length} characters.`);
      summary = responseText;
      transcription = 'N/A';

      if (!isPdf) {
        console.log('[AI Worker] Parsing Gemini response to separate Summary and Transcription blocks...');
        const splitIndex = responseText.toLowerCase().indexOf('transcript:');
        const alternateSplitIndex = responseText.toLowerCase().indexOf('## transcript');
        const targetIndex = splitIndex !== -1 ? splitIndex : (alternateSplitIndex !== -1 ? alternateSplitIndex : -1);

        if (targetIndex !== -1) {
          summary = responseText.substring(0, targetIndex).trim();
          transcription = responseText.substring(targetIndex).trim();
          console.log(`[AI Worker] Successfully split content. Summary length: ${summary.length}, Transcription length: ${transcription.length}`);
        } else {
          console.log('[AI Worker] Warning: Could not find explicit transcription separator in Gemini response. Leaving transcription as N/A.');
        }
      }
    }

    // 2. Authenticate and Execute Secure Callback to Web Backend
    // Target audience is derived from the base origin of your callback endpoint
    const audience = new URL(callbackUrl).origin;
    console.log(`[AI Worker] Creating authenticated OIDC client for backend audience: ${audience}`);

    const client = await auth.getIdTokenClient(audience);

    console.log(`[AI Worker] Dispatching secure POST callback to Web Backend via Auth Client: ${callbackUrl}`);
    const response = await client.request({
      url: callbackUrl,
      method: 'POST',
      data: {
        userId,
        summary,
        transcription,
      },
      headers: {
        'Content-Type': 'application/json',
      },
    });

    console.log(`[AI Worker] Callback completed successfully. HTTP status response: ${response.status}`);
    return res.status(200).json({
      status: 'success',
      message: 'AI Processing and Callback completed successfully',
      fileId,
    });

  } catch (error: any) {
    console.error('[AI Worker] Critical Error during processing or callback:');
    console.error(`  - Message: ${error.message || error}`);

    // Google's underlying gaxios client attaches response logs to error.response
    if (error.response) {
      console.error(`  - Callback Response Error Status: ${error.response.status}`);
      console.error('  - Callback Response Error Data:', JSON.stringify(error.response.data));
    }
    if (error.stack) {
      console.error(error.stack);
    }

    // Return a 500 error so that Cloud Tasks automatically schedules a retry
    return res.status(500).json({
      status: 'error',
      message: error.message || 'Internal processing error',
    });
  }
});

app.listen(port, () => {
  console.log(`[AI Worker] Server is running on port: ${port}`);
});