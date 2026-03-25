CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
	user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	first_name TEXT NOT NULL,
	last_name TEXT NOT NULL,
	email TEXT NOT NULL UNIQUE,
	phone_number TEXT,
	city TEXT,
	zip_code TEXT,
	password_hash TEXT NOT NULL,
	password_salt TEXT NOT NULL,
	is_verified BOOLEAN NOT NULL DEFAULT FALSE,
	created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS profile (
	profile_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
	user_id UUID NOT NULL UNIQUE,
	age INTEGER,
	lifestyle_parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
	personal_details JSONB NOT NULL DEFAULT '{}'::jsonb,
	updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS daily_inputs (
	entry_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
	user_id UUID NOT NULL,
	input_date DATE NOT NULL,
	submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	answers_json JSONB NOT NULL,
	prediction_json JSONB NOT NULL,
	cluster INTEGER,
	FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
	UNIQUE (user_id, input_date)
);
